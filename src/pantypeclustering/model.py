import torch
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
)
from torch import Tensor, nn
from torch.distributions import Normal

from pantypeclustering.architectures.conv import Decoder, Encoder, PriorGenerator


class GMVAE(nn.Module):
    def __init__(
        self,
        x_size: int,
        z1_size: int,
        z2_size: int,
        hidden_size: int,
        number_of_mixtures: int,
        mc: int,
        continuous: bool,
        lambda_threshold: float,
    ):
        super().__init__()  # type: ignore

        self.encoder = Encoder(
            hidden_size=hidden_size,
            z1_size=z1_size,
            z2_size=z2_size,
        )

        self.decoder = Decoder(
            z1_size=z1_size,
            hidden_size=hidden_size,
        )

        self.priorgenerator = PriorGenerator(
            z2_size=z2_size,
            hidden_size=hidden_size,
            z1_size=z1_size,
            number_of_mixtures=number_of_mixtures,
        )

        self.mc = mc
        self.cont = continuous
        self.lambda_threshold = lambda_threshold
        self.x_size, self.y_size, self.w_size = z1_size, x_size, z2_size
        self.hidden_size = hidden_size
        self.number_of_mixtures = number_of_mixtures

    def forward(self, x: Tensor) -> Tensor:
        (mean_z1, logvar_z1), (mean_z2, logvar_z2) = self.encoder(x)

        logvar_z1 = torch.clamp(logvar_z1, min=-30.0, max=20.0)
        logvar_z2 = torch.clamp(logvar_z2, min=-30.0, max=20.0)

        z1_dist = Normal(mean_z1, torch.exp(0.5 * logvar_z1))
        z2_dist = Normal(mean_z2, torch.exp(0.5 * logvar_z2))

        z1_sample = z1_dist.rsample((self.mc,))
        z2_sample = z2_dist.rsample((self.mc,))

        _, self.batch_size, _ = z1_sample.shape  # M x B x x_size

        z1_sample = z1_sample.view(self.batch_size * self.mc, self.x_size)
        z2_sample = z2_sample.view(self.batch_size * self.mc, self.w_size)

        x_recon = self.decoder(z1_sample)
        x_recon = x_recon.view(self.mc, self.batch_size, 1, 28, 28)

        (means, logvars) = self.priorgenerator(z2_sample)

        z1_sample = z1_sample.view(self.mc, self.batch_size, self.x_size)
        z2_sample = z2_sample.view(self.mc, self.batch_size, self.w_size)

        means = torch.stack([mean.view(self.mc, self.batch_size, self.x_size) for mean in means])
        logvars = torch.stack(
            [logvar.view(self.mc, self.batch_size, self.x_size) for logvar in logvars],
        )

        # Compute log-likelihood
        p_y = self.gaussian_mixture(z1_sample=z1_sample, means=means, logvars=logvars)

        # 1.) reconstruction loss
        recon_loss = self.reconstruction_loss(x=x, x_recon=x_recon)

        # 2.) E_z_w [KL(q(x)||p(x|z,w))]
        exp_kl = self.conditional_kl_term(
            mean_z1=mean_z1,
            logvar_z1=logvar_z1,
            means=means,
            logvars=logvars,
            p_y=p_y,
        )

        # 3.) KL( q(w) || P(w) )
        vae_kld_loss = self.kl_term(mean_z2, logvar_z2)

        # 4.) CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
        yloss = self.y_loss(p_y)

        total_loss = recon_loss + exp_kl + vae_kld_loss + yloss

        return total_loss, (x, x_recon)

    def gaussian_mixture(self, z1_sample: Tensor, means: Tensor, logvars: Tensor) -> Tensor:
        dist = Normal(means, torch.exp(0.5 * logvars))
        # z1_sample is [M, B, D], need [K, M, B, D] to match means
        num_mixtures = means.shape[0]
        z1_sample_expanded = z1_sample.unsqueeze(0).expand(num_mixtures, -1, -1, -1)
        llh = dist.log_prob(z1_sample_expanded)

        # [K, M, B, D] -> [K, M, B]
        llh = llh.sum(-1)  # sum over independent log gauss prob in x-dimension

        # [K, M, B]
        return llh.softmax(dim=0)  # softmax over K

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        recon_criterion = nn.MSELoss(reduction="sum") if self.cont else nn.BCELoss(reduction="sum")

        return recon_criterion(
            x_recon,
            x.unsqueeze(0).expand_as(x_recon),
        ) / (self.mc * self.batch_size)  # normalize by MC samples and batch size

    def conditional_kl_term(
        self,
        mean_z1: Tensor,
        logvar_z1: Tensor,
        means: Tensor,
        logvars: Tensor,
        p_y: Tensor,
    ) -> Tensor:
        var_x = logvar_z1.exp()
        var_k = logvars.exp()

        kl = 0.5 * (
            logvars
            - logvar_z1.unsqueeze(0)
            + (var_x.unsqueeze(0) + (mean_z1.unsqueeze(0) - means) ** 2) / var_k
            - 1
        )  # [K, M, B, D]

        kl = kl.sum(-1)  # [K, M, B]
        return (p_y * kl).mean(
            dim=1,
        ).sum() / self.batch_size  # avg MC, sum K, normalize by batch size

    def kl_term(self, mean_z2: Tensor, logvar_z2: Tensor) -> Tensor:
        return (
            -0.5
            * torch.sum(
                1 + logvar_z2 - mean_z2.pow(2) - logvar_z2.exp(),
            )
            / self.batch_size
        )  # normalize by batch size

    def y_loss(self, p_y: Tensor) -> Tensor:
        conditional_entropy = torch.log(p_y + 1e-10) * p_y
        conditional_entropy = -conditional_entropy.sum() / (self.mc * self.batch_size)
        # normalize by MC samples and batch size

        # yloss should encourage diversity in cluster assignments
        # The log(1/w_dim) erm is the entropy of uniform prior, should not scale with B
        yloss = -conditional_entropy - torch.log(torch.tensor(1.0) / self.w_size)
        return torch.max(yloss, torch.tensor(self.lambda_threshold))

    def acc_evaluation(self, generated_label: Tensor, true_label: Tensor) -> Tensor:
        x_n = generated_label.argmax(0)
        labels = generated_label.argmax(1)

        cluster_labels = true_label[x_n]

        #  [N x K]
        #  [K]

        acc = torch.tensor(0)
        for k in range(self.number_of_mixtures):
            acc += ((labels == k) & (true_label == cluster_labels[k])).sum()

        acc = acc / len(generated_label)

        return acc

    def get_db_score(self, data_points: Tensor, test_class_probs: Tensor) -> float:
        labels = test_class_probs.argmax(1)
        if len(torch.unique(labels)) > 1:
            return davies_bouldin_score(data_points.numpy(), labels)
        return torch.nan
        # db_score = davies_bouldin_score(cluster_labels, )

    def get_adjust_rand(self, true_label: Tensor, test_class_probs: Tensor) -> float:
        labels = test_class_probs.argmax(1)

        return adjusted_rand_score(true_label, labels)

    def get_class_prob(self, x: Tensor) -> Tensor:
        (mean_z1, logvar_z1), (mean_z2, logvar_z2) = self.encoder(x)

        logvar_z1 = torch.clamp(logvar_z1, min=-30.0, max=20.0)
        logvar_z2 = torch.clamp(logvar_z2, min=-30.0, max=20.0)

        z1_dist = Normal(mean_z1, torch.exp(0.5 * logvar_z1))
        z2_dist = Normal(mean_z2, torch.exp(0.5 * logvar_z2))

        z1_sample = z1_dist.rsample((1,))
        z2_sample = z2_dist.rsample((1,))

        _, batch_size, z1_size = z1_sample.shape  # M x B x x_size
        _, _, z2_size = z2_sample.shape

        z1_sample = z1_sample.view(batch_size, z1_size)
        z2_sample = z2_sample.view(batch_size, z2_size)

        (means, logvars) = self.priorgenerator(z2_sample)

        means = torch.stack([mean.view(batch_size, z1_size) for mean in means])
        logvars = torch.stack([logvar.view(batch_size, z1_size) for logvar in logvars])

        # Compute log-likelihood
        dist = Normal(means, torch.exp(0.5 * logvars))
        # z1_sample is [B, D], need [K, B, D] to match means
        num_mixtures = means.shape[0]
        z1_sample_expanded = z1_sample.unsqueeze(0).expand(num_mixtures, -1, -1)
        llh = dist.log_prob(z1_sample_expanded)

        # [K, B, D] -> [K, B]
        llh = llh.sum(-1)  # sum over independent log gauss prob in x-dimension

        # [K, B]
        p_y = llh.softmax(dim=0)  # softmax over K

        return p_y

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input images y using the VAE model."""
        self.eval()
        with torch.no_grad():
            (mean_z1, _), (_, _) = self.encoder(x)
            y_mean = self.decoder(mean_z1)
            return y_mean
