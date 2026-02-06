import torch
from torch import Tensor, nn
from torch.distributions import Normal

from pantypeclustering.architectures.conv import PriorGenerator, Encoder, Decoder

from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,  # pyright: ignore[reportUnknownVariableType]
    adjusted_rand_score,
)


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

        self.recogniser = Encoder(
            hidden_size=hidden_size,
            z1_size=z1_size,
            z2_size=z2_size,
        )

        self.ygenerator = Decoder(
            z1_size=z1_size,
            hidden_size=hidden_size,
        )

        self.priorgenerator = PriorGenerator(
            z2_size=z2_size,
            hidden_size=hidden_size,
            z1_size=z1_size,
            number_of_mixtures=number_of_mixtures
        )

        self.mc = mc
        self.cont = continuous
        self.lambda_threshold = lambda_threshold
        self.x_size, self.y_size, self.w_size = z1_size, x_size, z2_size
        self.hidden_size = hidden_size
        self.number_of_mixtures = number_of_mixtures


    def forward(self, y: Tensor) -> Tensor:
        (mean_x, logvar_x), (mean_w, logvar_w) = self.recogniser(y)

        logvar_x = torch.clamp(logvar_x, min=-30.0, max=20.0)
        logvar_w = torch.clamp(logvar_w, min=-30.0, max=20.0)

        x_dist = Normal(mean_x, torch.exp(0.5 * logvar_x))
        w_dist = Normal(mean_w, torch.exp(0.5 * logvar_w))

        x_sample = x_dist.rsample((self.mc,))
        w_sample = w_dist.rsample((self.mc,))

        _, self.batch_size, _ = x_sample.shape  # M x B x x_size
 
        x_sample = x_sample.view(self.batch_size * self.mc, self.x_size)
        w_sample = w_sample.view(self.batch_size * self.mc, self.w_size)

        y_recon = self.ygenerator(x_sample)
        y_recon = y_recon.view(self.mc, self.batch_size, 1, 28, 28)

        (means, logvars) = self.priorgenerator(w_sample)

        x_sample = x_sample.view(self.mc, self.batch_size, self.x_size)
        w_sample = w_sample.view(self.mc, self.batch_size, self.w_size)

        means = torch.stack([mean.view(self.mc, self.batch_size, self.x_size) for mean in means])
        logvars = torch.stack([logvar.view(self.mc, self.batch_size, self.x_size) for logvar in logvars])

        # Compute log-likelihood
        p_z = self.gaussian_mixture(x_sample, means, logvars)

        # 1.) reconstruction loss
        recon_loss = self.reconstruction_loss(y=y, y_recon=y_recon)

        # 2.) E_z_w [KL(q(x)||p(x|z,w))]
        exp_kl = self.conditional_kl_term(
            mean_x=mean_x,
            logvar_x=logvar_x,
            means=means,
            logvars=logvars,
            p_z=p_z
        )

        # 3.) KL( q(w) || P(w) )
        vae_kld_loss = self.kl_term(mean_w, logvar_w)

        # 4.) CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
        zloss = self.z_loss(p_z)

        total_loss = recon_loss + exp_kl + vae_kld_loss + zloss


        return total_loss, (y, y_recon)

    def gaussian_mixture(self, x_sample: Tensor, means: Tensor, logvars: Tensor) -> Tensor:
        dist = Normal(means, torch.exp(0.5 * logvars))
        # x_sample is [M, B, D], need [K, M, B, D] to match means
        num_mixtures = means.shape[0]
        x_sample_expanded = x_sample.unsqueeze(0).expand(num_mixtures, -1, -1, -1)
        llh = dist.log_prob(x_sample_expanded)

        # [K, M, B, D] -> [K, M, B]
        llh = llh.sum(-1)  # sum over independent log gauss prob in x-dimension

        # [K, M, B]
        return llh.softmax(dim=0)  # softmax over K

    def reconstruction_loss(self, y: Tensor, y_recon: Tensor) -> Tensor:
        recon_criterion = nn.MSELoss(reduction='sum') if self.cont else nn.BCELoss(reduction='sum')

        return recon_criterion(
            y_recon,
            y.unsqueeze(0).expand_as(y_recon)
        ) / (self.mc * self.batch_size)  # normalize by MC samples and batch size

    def conditional_kl_term(
            self,
            mean_x: Tensor,
            logvar_x: Tensor,
            means: Tensor,
            logvars: Tensor,
            p_z: Tensor
        ) -> Tensor:
        var_x = logvar_x.exp()
        var_k = logvars.exp()

        kl = 0.5 * (
            logvars - logvar_x.unsqueeze(0)
            + (var_x.unsqueeze(0) + (mean_x.unsqueeze(0) - means) ** 2) / var_k
            - 1
        )  # [K, M, B, D]

        kl = kl.sum(-1)  # [K, M, B]
        return (p_z * kl).mean(dim=1).sum() / self.batch_size  # avg MC, sum K, normalize by batch size

    def kl_term(self, mean_w: Tensor, logvar_w: Tensor) -> Tensor:
        return -0.5 * torch.sum(
            1 + logvar_w - mean_w.pow(2) - logvar_w.exp()
        ) / self.batch_size  # normalize by batch size

    def z_loss(self, p_z: Tensor) -> Tensor:
        conditional_entropy = torch.log(p_z + 1e-10) * p_z
        conditional_entropy = -conditional_entropy.sum() / (self.mc * self.batch_size)
        # normalize by MC samples and batch size

        # zloss should encourage diversity in cluster assignments
        # The log(1/w_dim) erm is the entropy of uniform prior, should not scale with B
        zloss = -conditional_entropy - torch.log(torch.tensor(1.0) / self.w_size)
        return torch.max(zloss, torch.tensor(self.lambda_threshold))


    def acc_evaluation(self, generated_label: Tensor, true_label: Tensor) -> Tensor:

        x_n = generated_label.argmax(0)
        labels = generated_label.argmax(1)

        cluster_labels = true_label[x_n]

        #  [N x K]
        #  [K]

        acc = torch.tensor(0)
        for k in range(self.number_of_mixtures):
            acc += ((labels == k) & (true_label == cluster_labels[k])).sum()

        acc = acc/len(generated_label)

        return acc


    def get_db_score(self, data_points: Tensor, test_class_probs: Tensor) -> float:
        labels = test_class_probs.argmax(1)
        if len(torch.unique(labels)) > 1:
            return davies_bouldin_score(data_points, labels)
        return torch.nan
        #db_score = davies_bouldin_score(cluster_labels, )

    def get_adjust_rand(self, true_label: Tensor, test_class_probs: Tensor) -> float:
        labels = test_class_probs.argmax(1)

        return adjusted_rand_score(true_label, labels)



    def get_class_prob(self, y: Tensor) -> Tensor:
        (mean_x, logvar_x), (mean_w, logvar_w) = self.recogniser(y)

        logvar_x = torch.clamp(logvar_x, min=-30.0, max=20.0)
        logvar_w = torch.clamp(logvar_w, min=-30.0, max=20.0)

        x_dist = Normal(mean_x, torch.exp(0.5 * logvar_x))
        w_dist = Normal(mean_w, torch.exp(0.5 * logvar_w))

        x_sample = x_dist.rsample((1,))
        w_sample = w_dist.rsample((1,))

        _, batch_size, x_size = x_sample.shape  # M x B x x_size
        _, _, w_size = w_sample.shape  # M x B x w_size

        x_sample = x_sample.view(batch_size, x_size)
        w_sample = w_sample.view(batch_size, w_size)

        (means, logvars) = self.priorgenerator(w_sample)

        means = torch.stack([mean.view(batch_size, x_size) for mean in means])
        logvars = torch.stack([logvar.view(batch_size, x_size) for logvar in logvars])

        # Compute log-likelihood
        dist = Normal(means, torch.exp(0.5 * logvars))
        # x_sample is [B, D], need [K, B, D] to match means
        num_mixtures = means.shape[0]
        x_sample_expanded = x_sample.unsqueeze(0).expand(num_mixtures, -1, -1)
        llh = dist.log_prob(x_sample_expanded)

        # [K, B, D] -> [K, B]
        llh = llh.sum(-1)  # sum over independent log gauss prob in x-dimension

        # [K, B]
        p_z = llh.softmax(dim=0)  # softmax over K

        return p_z

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        """Reconstruct input images y using the VAE model."""
        self.eval()
        with torch.no_grad():
            (mean_x, logvar_x), (mean_w, logvar_w) = self.recogniser(y)
            y_mean = self.ygenerator(mean_x)
            return y_mean



