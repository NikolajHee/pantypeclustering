import torch
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from torch import Tensor, nn
from torch.distributions import Normal, Categorical

import pytorch_lightning as pl

from pantypeclustering.architectures.conv import Decoder, Encoder, PriorGenerator

NUMBER_OF_CLASSES = 10
class GMVAE(pl.LightningModule):
    """Gaussian Mixture VAE for unsupervised clustering of images."""

    def __init__(
            self,
            learning_rate: float,
            x_size: int,
            z1_size: int,
            z2_size: int,
            hidden_size: int,
            number_of_mixtures: int,
            mc: int,
            continuous: bool,
            lambda_threshold: float,
            N: int,
            seed: int,
    ):
        super().__init__()  # type: ignore

        self.encoder = Encoder(
            hidden_size=hidden_size,
            x_size=z1_size,
            w_size=z2_size,
        )

        self.decoder = Decoder(
            input_size=z1_size,
            hidden_size=hidden_size,
        )

        self.priorgenerator = PriorGenerator(
            input_size=z2_size,
            hidden_size=hidden_size,
            output_size=z1_size,
            number_of_mixtures=number_of_mixtures
        )

        self.learning_rate = learning_rate
        self.mc = mc
        self.cont = continuous
        self.lambda_threshold = lambda_threshold
        self.z1_size, self.x_size, self.z2_size = z1_size, x_size, z2_size
        self.hidden_size = hidden_size
        self.number_of_mixtures = number_of_mixtures
        self.N = N

        self.acc: list[float] = []
        self.seed = seed

        self.prior_logits = nn.Parameter(torch.zeros(number_of_mixtures))

    def forward(self, x: Tensor) -> Tensor:
        """Compute ELBO loss and return (original, reconstructed) images."""
        (mean_z1, logvar_z1), (mean_z2, logvar_z2) = self.encoder(x)

        logvar_z1 = torch.clamp(logvar_z1, min=-30.0, max=20.0)
        logvar_z2 = torch.clamp(logvar_z2, min=-30.0, max=20.0)

        z1_dist = Normal(mean_z1, torch.exp(0.5 * logvar_z1))
        z2_dist = Normal(mean_z2, torch.exp(0.5 * logvar_z2))

        z1_sample = z1_dist.rsample((self.mc,))
        z2_sample = z2_dist.rsample((self.mc,))

        _, self.batch_size, _ = z1_sample.shape  # M x B x x_size

        z1_sample = z1_sample.view(self.batch_size * self.mc, self.z1_size)
        z2_sample = z2_sample.view(self.batch_size * self.mc, self.z2_size)

        x_recon = self.decoder(z1_sample)
        x_recon = x_recon.view(self.mc, self.batch_size, 1, 28, 28)

        (means, logvars) = self.priorgenerator(z2_sample)

        z1_sample = z1_sample.view(self.mc, self.batch_size, self.z1_size)
        z2_sample = z2_sample.view(self.mc, self.batch_size, self.z2_size)

        means = torch.stack([mean.view(self.mc, self.batch_size, self.z1_size) for mean in means])
        logvars = torch.stack([logvar.view(self.mc, self.batch_size, self.z1_size) for logvar in logvars])

        # Compute log-likelihood
        p_y = self.gaussian_mixture(z1_sample, means, logvars)

        # 1.) reconstruction loss
        recon_loss = self.reconstruction_loss(x=x, x_recon=x_recon)

        # 2.) E_z_w [KL(q(x)||p(x|z,w))]
        exp_kl = self.conditional_kl_term(
            mean_z1=mean_z1,
            logvar_z1=logvar_z1,
            means=means,
            logvars=logvars,
            p_y=p_y
        )

        # 3.) KL( q(w) || P(w) )
        vae_kld_loss = self.kl_term(mean_z2, logvar_z2)

        # 4.) CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
        yloss = self.y_loss(p_y)

        total_loss = recon_loss + exp_kl + vae_kld_loss + yloss

        return total_loss, (x, x_recon)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        images, _ = batch
        loss, _ = self.forward(images)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        images, label = batch
        loss, _ = self.forward(images)
        self.log('val_loss', loss)

        # Save test data for evaluation
        start_idx = batch_idx * len(label)
        end_idx = start_idx + len(label)
        self.test_class_probs[start_idx:end_idx] = self.get_class_prob(images).T
        self.test_label[start_idx:end_idx] = label
        (mean_z1, _), (_, _) = self.encoder(images)
        self.test_z1[start_idx:end_idx] = mean_z1

    def on_validation_epoch_start(self) -> None:
        self.reset_test_saves()

    def on_validation_epoch_end(self) -> None:
        """Evaluate clustering metrics at the end of validation epoch."""

        acc = self.acc_evaluation(self.test_class_probs, self.test_label)
        db_score = self.get_db_score(self.test_z1, self.test_class_probs)
        adj_rand = self.get_adjust_rand(self.test_label, self.test_class_probs)

        self.acc.append(acc.item())

        self.log('val_acc', acc, prog_bar=True)
        self.log('val_db', db_score)
        self.log('val_adj_rand', adj_rand)

    def on_fit_end(self) -> None:
        pass
        #torch.save(self.acc, f'acc_history_alternative_{self.seed}.pt')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=(self.learning_rate))

    def gaussian_mixture(self, z1_sample: Tensor, means: Tensor, logvars: Tensor) -> Tensor:
        """Compute mixture component posterior p(y|z1, z2)."""
        dist = Normal(means, torch.exp(0.5 * logvars))
        # z1_sample is [M, B, D], need [K, M, B, D] to match means
        num_mixtures = means.shape[0]
        z1_sample_expanded = z1_sample.unsqueeze(0).expand(num_mixtures, -1, -1, -1)
        llh = dist.log_prob(z1_sample_expanded)

        llh = llh.sum(-1)  # sum over independent log gauss prob in x-dimension

        log_p_y = torch.log_softmax(llh, dim=0)
        p_y = log_p_y.exp()

        return p_y

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """MSE or BCE loss for reconstruction."""
        recon_criterion = nn.MSELoss(reduction='sum') if self.cont else nn.BCELoss(reduction='sum')

        return recon_criterion(
            x_recon,
            x.unsqueeze(0).expand_as(x_recon)
        ) / (self.mc * self.batch_size)  # normalize by MC samples and batch size

    def conditional_kl_term(
            self,
            mean_z1: Tensor,
            logvar_z1: Tensor,
            means: Tensor,
            logvars: Tensor,
            p_y: Tensor,
    ) -> Tensor:
        """Conditional KL E[KL(q(z1|x)||p(z1|z2))]."""
        var_z1 = logvar_z1.exp()
        var_k = logvars.exp()

        kl = 0.5 * (
            logvars - logvar_z1.unsqueeze(0)
            + (var_z1.unsqueeze(0) + (mean_z1.unsqueeze(0) - means) ** 2) / var_k
            - 1
        )  # [K, M, B, D]

        kl = kl.sum(-1)  # [K, M, B]
        return (p_y * kl).mean(dim=1).sum() / self.batch_size  # avg MC, sum K, normalize by batch size

    def kl_term(self, mean_z2: Tensor, logvar_z2: Tensor) -> Tensor:
        """KL divergence from encoder posterior to standard normal prior."""
        return -0.5 * torch.sum(
            1 + logvar_z2 - mean_z2.pow(2) - logvar_z2.exp()
        ) / self.batch_size  # normalize by batch size

    def _y_loss(self, p_y: Tensor) -> Tensor:
        """Information-theoretic clustering loss (min entropy, free bits)."""
        conditional_entropy = torch.log(p_y + 1e-10) * p_y
        conditional_entropy = -conditional_entropy.sum() / (self.mc * self.batch_size)

        yloss = -conditional_entropy - torch.log(torch.tensor(1.0) / self.number_of_mixtures)
        return yloss

    def _y_loss_alternative(self, p_y: Tensor) -> Tensor:
        """Alternative clustering loss: KL divergence from uniform distribution."""

        # prior_probs = torch.softmax(self.prior_logits, dim=0)
        # discrete_kl = p_y * (torch.log(p_y + 1e-10)).sum() / (self.mc * self.batch_size) \
        #             - p_y * torch.log(prior_probs + 1e-10).sum()
        # #- torch.log(torch.tensor(1.0 / self.number_of_mixtures))
        # return discrete_kl

        discrete_kl = p_y * (torch.log(self.number_of_mixtures * p_y + 1e-10))
        return discrete_kl.sum() / (self.mc * self.batch_size)

        # variable prior didn't work well, so using fixed uniform prior instead
        #prior_probs = torch.softmax(self.prior_logits, dim=0)  # [K]
        #log_prior = torch.log(prior_probs + 1e-10).view(-1, 1, 1)

        #log_p_y = torch.log(p_y + 1e-10)

        #kl = p_y * (log_p_y - log_prior)  # [K, M, B]

        # return kl.sum() / (self.mc * self.batch_size)

    def y_loss(self, p_y: Tensor) -> Tensor:
        """Wrapper for clustering loss with option for free bits."""
        yloss = self._y_loss_alternative(p_y)
        return torch.max(yloss, torch.tensor(self.lambda_threshold))

    def reset_test_saves(self) -> None:
        """Reset saved test data for evaluation."""
        self.test_class_probs = torch.zeros((self.N, self.number_of_mixtures))
        self.test_label = torch.zeros((self.N))
        self.test_z1 = torch.zeros((self.N, self.z1_size))

    def acc_evaluation(self, generated_label: Tensor, true_label: Tensor) -> Tensor:
        """Clustering accuracy via best label permutation."""
        x_n = generated_label.argmax(0)
        labels = generated_label.argmax(1)

        cluster_labels = true_label[x_n]

        #  [N x K]
        #  [K]

        acc = torch.tensor(0)
        for k in range(self.number_of_mixtures):
            acc += ((labels == k) & (true_label == cluster_labels[k])).sum()

        return acc / len(generated_label)

    def hungarian_accuracy(self, generated_label: Tensor, true_label: Tensor) -> Tensor:
        """Clustering accuracy via Hungarian algorithm for optimal label matching."""
        from scipy.optimize import linear_sum_assignment
        labels = generated_label.argmax(1)

        C = torch.zeros(self.number_of_mixtures, NUMBER_OF_CLASSES, dtype=torch.int64)

        for i in range(self.number_of_mixtures):
            C[labels[i], true_label[i]] += 1

        linear_sum_assignment()

    def get_db_score(self, data_points: Tensor, test_class_probs: Tensor) -> float:
        """Davies-Bouldin index for cluster compactness."""
        labels = test_class_probs.argmax(1)
        if len(torch.unique(labels)) > 1:
            return davies_bouldin_score(data_points, labels)
        return torch.nan

    def get_adjust_rand(self, true_label: Tensor, test_class_probs: Tensor) -> float:
        """Adjusted Rand index between predicted and true labels."""
        labels = test_class_probs.argmax(1)

        return adjusted_rand_score(true_label, labels)

    def get_class_prob(self, x: Tensor) -> Tensor:
        """Posterior cluster assignment probabilities for input batch."""
        (mean_z1, logvar_z1), (mean_z2, logvar_z2) = self.encoder(x)

        logvar_z1 = torch.clamp(logvar_z1, min=-30.0, max=20.0)
        logvar_z2 = torch.clamp(logvar_z2, min=-30.0, max=20.0)

        z1_dist = Normal(mean_z1, torch.exp(0.5 * logvar_z1))
        z2_dist = Normal(mean_z2, torch.exp(0.5 * logvar_z2))

        z1_sample = z1_dist.rsample((1,))
        z2_sample = z2_dist.rsample((1,))

        _, batch_size, z1_size = z1_sample.shape  # M x B x x_size
        _, _, z2_size = z2_sample.shape  # M x B x w_size

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
        """Decode latent mean to reconstructed images."""
        self.eval()
        with torch.no_grad():
            (mean_z1, logvar_z1), (mean_z2, logvar_z2) = self.encoder(x)
            x_mean = self.decoder(mean_z1)
            return x_mean

