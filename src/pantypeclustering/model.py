

import torch
from torch import Tensor, nn
from torch.distributions import LogNormal, Normal

from pantypeclustering.architectures.conv import Recogniser, YGenerator, PriorGenerator
from pantypeclustering.dataloader import get_mnist_dataloaders


class GMVAE(nn.Module):
    def __init__(
            self,
            recogniser: Recogniser,
            ygenerator: YGenerator,
            priorgenerator: PriorGenerator,
            mc: int,
            continuous: bool,
    ):
        super().__init__()
        self.recogniser = recogniser
        self.ygenerator = ygenerator
        self.priorgenerator = priorgenerator
        self.mc = mc
        self.cont = continuous


    def forward(self, batch: Tensor):
        y = batch
        (mean_x, logVar_x), (mean_w, logVar_w) = self.recogniser(y)

        logVar_x = torch.clamp(logVar_x, min=-30.0, max=20.0)
        logVar_w = torch.clamp(logVar_w, min=-30.0, max=20.0)

        x_dist = Normal(mean_x, torch.exp(0.5 * logVar_x))
        w_dist = Normal(mean_w, torch.exp(0.5 * logVar_w))

        x_sample = x_dist.rsample((self.mc,))
        w_sample = w_dist.rsample((self.mc,))

        _, B, x_size = x_sample.shape # M x B x x_size
        _, _, w_size = w_sample.shape # M x B x w_size

        x_sample = x_sample.view(B*self.mc, x_size)
        w_sample = w_sample.view(B*self.mc, w_size)

        y_recon = self.ygenerator(x_sample)
        y_recon = y_recon.view(self.mc, B, 1, 28, 28)

        (means, logvars) = self.priorgenerator(w_sample)

        x_sample = x_sample.view(self.mc, B, x_size)
        w_sample = w_sample.view(self.mc, B, w_size)

        means = torch.stack([mean.view(self.mc, B, x_size) for mean in means])
        logvars = torch.stack([logvar.view(self.mc, B, x_size) for logvar in logvars])

        # llh
        dist = Normal(means, torch.exp(0.5 * logvars))
        # x_sample is [M, B, D], need [K, M, B, D] to match means
        x_sample_expanded = x_sample.unsqueeze(0).expand(10, -1, -1, -1)
        llh = dist.log_prob(x_sample_expanded)

        # [K, M, B, D] -> [K, M, B]
        llh = llh.sum(-1) # sum over independent log gauss prob in x-dimension
        
        # [K, M, B]
        p_z = llh.softmax(dim=0) # softmax over K

        # 1.) reconstruction loss
        if self.cont:
            recon_criterion = nn.MSELoss(reduction='sum')
        else:
            recon_criterion = nn.BCELoss(reduction='sum')

        recon_loss = recon_criterion(
            y_recon,
            y.unsqueeze(0).expand_as(y_recon)
        ) / (self.mc * B)  # normalize by MC samples and batch size


        # 2.) E_z_w [KL(q(x)||p(x|z,w))]
        K, mc, B, D = means.shape
        B, D2 = mean_w.shape

        #mean_x = mean_x.repeat(mc, 1, 1)
        #logVar_x = logVar_x.repeat(mc, 1, 1)

        var_x = logVar_x.exp()
        var_k = logvars.exp()

        kl = 0.5 * (
            logvars - logVar_x.unsqueeze(0)
            + (var_x.unsqueeze(0) + (mean_x.unsqueeze(0) - means)**2) / var_k
            - 1
        )  # [K, M, B, D]

        kl = kl.sum(-1)  # [K, M, B]
        exp_kl = (p_z * kl).mean(dim=1).sum() / B  # avg MC, sum K, normalize by batch size

        
        
        # 3.) KL( q(w) || P(w) )
        vae_kld_loss = -0.5 * torch.sum(
                1 + logVar_w - mean_w.pow(2) - logVar_w.exp()
            ) / B  # normalize by batch size

        # 4.)  CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]

        CV = torch.log(p_z + 1e-10) * p_z
        CV = - CV.sum() / (self.mc * B)  # normalize by MC samples and batch size

        _lambda = 0.5
        # zloss should encourage diversity in cluster assignments
        # The log(1/D2) term is the entropy of uniform prior, should not scale with B
        zloss = - CV - torch.log(torch.tensor(1.0) / D2)
        zloss = torch.max(zloss, torch.tensor(_lambda))

        total_loss = recon_loss + exp_kl + vae_kld_loss + zloss

        # print(f"recon_loss: {recon_loss}")
        # print(f"exp_kl_loss: {exp_kl_loss}")
        # print(f"vae_kld_loss: {vae_kld_loss}")
        # print(f"zloss: {zloss}")


        return total_loss, (y, y_recon)

    def acc_evaluation(self, generated_label: torch.Tensor, true_label: torch.Tensor):

        x_n = generated_label.argmax(0)
        labels = generated_label.argmax(1)

        cluster_labels = true_label[x_n]

        #  [N x K]
        #  [K] 

        acc = 0
        for k in range(10):
            acc += ((labels == k) & (true_label == cluster_labels[k])).sum()
        
        acc = acc/len(generated_label)

        return acc

    def get_class_prob(self, y: torch.Tensor):
        (mean_x, logVar_x), (mean_w, logVar_w) = self.recogniser(y)

        logVar_x = torch.clamp(logVar_x, min=-30.0, max=20.0)
        logVar_w = torch.clamp(logVar_w, min=-30.0, max=20.0)

        x_dist = Normal(mean_x, torch.exp(0.5 * logVar_x))
        w_dist = Normal(mean_w, torch.exp(0.5 * logVar_w))

        x_sample = x_dist.rsample((1,))
        w_sample = w_dist.rsample((1,))

        _, B, x_size = x_sample.shape # M x B x x_size
        _, _, w_size = w_sample.shape # M x B x w_size

        x_sample = x_sample.view(B, x_size)
        w_sample = w_sample.view(B, w_size)

        (means, logvars) = self.priorgenerator(w_sample)

        x_sample = x_sample.view(B, x_size)
        w_sample = w_sample.view(B, w_size)

        means = torch.stack([mean.view(B, x_size) for mean in means])
        logvars = torch.stack([logvar.view(B, x_size) for logvar in logvars])

        # llh
        dist = Normal(means, torch.exp(0.5 * logvars))
        # x_sample is [B, D], need [K, B, D] to match means
        x_sample_expanded = x_sample.unsqueeze(0).expand(10, -1, -1)
        llh = dist.log_prob(x_sample_expanded)

        # [K, B, D] -> [K, B]
        llh = llh.sum(-1) # sum over independent log gauss prob in x-dimension
        
        # [K, M, B]
        p_z = llh.softmax(dim=0) # softmax over K
    
        return p_z
    
    


if __name__ == "__main__":


    input_size = 28*28
    hidden_size = 16
    x_size = 8
    w_size = 4
    number_of_mixtures=10

    recog = Recogniser(input_size=input_size,
                       hidden_size=hidden_size,
                       x_size=x_size,
                       w_size=w_size,
                       number_of_mixtures=number_of_mixtures)
    
    ygen = YGenerator(input_size=x_size,
                      hidden_size=hidden_size,
                      output_size=input_size,
                      continuous=True)
    
    priorgen = PriorGenerator(input_size=w_size,
                              hidden_size=hidden_size,
                              output_size=x_size,
                              number_of_mixtures=number_of_mixtures)

    model = GMVAE(recogniser=recog,
                  ygenerator=ygen,
                  priorgenerator=priorgen,
                  mc=5,
                  continuous=True)

    train_loader, test_loader = get_mnist_dataloaders(batch_size=256, binarize=False)

    some_iter = iter(train_loader)

    batch = next(some_iter)
    
    model.forward(batch[0])



