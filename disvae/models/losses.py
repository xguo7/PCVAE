"""
Module containing all vae losses.
"""
import abc
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from .discriminator import Discriminator
from disvae.utils.math import (log_density_gaussian, log_importance_weight_matrix,
                               matrix_log_density_gaussian)


LOSSES = ["VAE", "betaH", "betaB", "factor", "btcvae","btcvae_property", "btcvae_property_tc", "semivae_property", "csvae_property"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


# TO-DO: clean n_data and device
def get_loss_f(loss_name, **kwargs_parse):
    """Return the correct loss function given the argparse arguments."""
    kwargs_all = dict(rec_dist=kwargs_parse["rec_dist"],
                      steps_anneal=kwargs_parse["reg_anneal"])
    if loss_name == "betaH":
        return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)
    elif loss_name == "VAE":
        return BetaHLoss(beta=1, **kwargs_all)
    elif loss_name == "betaB":
        return BetaBLoss(C_init=kwargs_parse["betaB_initC"],
                         C_fin=kwargs_parse["betaB_finC"],
                         gamma=kwargs_parse["betaB_G"],
                         **kwargs_all)
    elif loss_name == "factor":
        return FactorKLoss(kwargs_parse["device"],
                           gamma=kwargs_parse["factor_G"],
                           disc_kwargs=dict(latent_dim=kwargs_parse["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs_parse["lr_disc"], betas=(0.5, 0.9)),
                           **kwargs_all)
    elif loss_name == "btcvae":
        return BtcvaeLoss(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)
    elif loss_name == "btcvae_property":
        return BtcvaeLoss_property(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all) 
        
    elif loss_name == "btcvae_property_tc":
        return BtcvaeLoss_property(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)  

    elif loss_name == "semivae_property":
        return SemivaeLoss_property(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)  
    
    elif loss_name == "csvae_property":
        return CsvaeLoss_property(kwargs_parse["n_data"],
                          alpha=kwargs_parse["btcvae_A"],
                          beta=kwargs_parse["btcvae_B"],
                          gamma=kwargs_parse["btcvae_G"],
                          **kwargs_all)         
    else:
        assert loss_name not in LOSSES
        raise ValueError("Uknown loss : {}".format(loss_name))


class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        Every how many steps to recorsd the loss.

    rec_dist: {"bernoulli", "gaussian", "laplace"}, optional
        Reconstruction distribution istribution of the likelihood on the each pixel.
        Implicitely defines the reconstruction loss. Bernoulli corresponds to a
        binary cross entropy (bse), Gaussian corresponds to MSE, Laplace
        corresponds to L1.

    steps_anneal: nool, optional
        Number of annealing steps where gradually adding the regularisation.
    """

    def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.rec_dist = rec_dist
        self.steps_anneal = steps_anneal

    @abc.abstractmethod
    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Shape : (batch_size, n_chan,
            height, width).

        recon_data : torch.Tensor
            Reconstructed data. Shape : (batch_size, n_chan, height, width).

        latent_dist : tuple of torch.tensor
            sufficient statistics of the latent dimension. E.g. for gaussian
            (mean, log_var) each of shape : (batch_size, latent_dim).

        is_train : bool
            Whether currently in train mode.

        storer : dict
            Dictionary in which to store important variables for vizualisation.

        kwargs:
            Loss specific arguments
        """

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 1:
            storer = storer
        else:
            storer = None

        return storer


class BetaHLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)
        loss = rec_loss + anneal_reg * (self.beta * kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


class BetaBLoss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
    """

    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
        storer = self._pre_call(is_train, storer)

        rec_loss = _reconstruction_loss(data, recon_data,
                                        storer=storer,
                                        distribution=self.rec_dist)
        kl_loss = _kl_normal_loss(*latent_dist, storer)

        C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
             if is_train else self.C_fin)

        loss = rec_loss + self.gamma * (kl_loss - C).abs()

        if storer is not None:
            storer['loss'].append(loss.item())

        return loss


        
class FactorKLoss(BaseLoss):
    """
    Compute the Factor-VAE loss as per Algorithm 2 of [1]

    Parameters
    ----------
    device : torch.device

    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.

    discriminator : disvae.discriminator.Discriminator

    optimizer_d : torch.optim

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """

    def __init__(self, device,
                 gamma=10.,
                 disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.discriminator = Discriminator(**disc_kwargs).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), **optim_kwargs)

    def __call__(self, *args, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, optimizer, storer):
        storer = self._pre_call(model.training, storer)

        # factor-vae split data into two batches. In the paper they sample 2 batches
        batch_size = data.size(dim=0)
        half_batch_size = batch_size // 2
        data = data.split(half_batch_size)
        data1 = data[0]
        data2 = data[1]

        # Factor VAE Loss
        recon_batch, latent_dist, latent_sample1 = model(data1)
        rec_loss = _reconstruction_loss(data1, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)

        kl_loss = _kl_normal_loss(*latent_dist, storer)

        d_z = self.discriminator(latent_sample1)
        # We want log(p_true/p_false). If not using logisitc regression but softmax
        # then p_true = exp(logit_true) / Z; p_false = exp(logit_false) / Z
        # so log(p_true/p_false) = logit_true - logit_false
        tc_loss = (d_z[:, 0] - d_z[:, 1]).mean()
        # with sigmoid (not good results) should be `tc_loss = (2 * d_z.flatten()).mean()`

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if model.training else 1)
        vae_loss = rec_loss + kl_loss + anneal_reg * self.gamma * tc_loss

        if storer is not None:
            storer['loss'].append(vae_loss.item())
            storer['tc_loss'].append(tc_loss.item())

        if not model.training:
            # don't backprop if evaluating
            return vae_loss

        # Run VAE optimizer
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Discriminator Loss
        # Get second sample of latent distribution
        latent_sample2 = model.sample_latent(data2)
        z_perm = _permute_dims(latent_sample2).detach()
        d_z_perm = self.discriminator(z_perm)

        # Calculate total correlation loss
        # for cross entropy the target is the index => need to be long and says
        # that it's first output for d_z and second for perm
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        d_tc_loss = 0.5 * (F.cross_entropy(d_z, zeros) + F.cross_entropy(d_z_perm, ones))
        # with sigmoid would be :
        # d_tc_loss = 0.5 * (self.bce(d_z.flatten(), ones) + self.bce(d_z_perm.flatten(), 1 - ones))

        # TO-DO: check ifshould also anneals discriminator if not becomes too good ???
        #d_tc_loss = anneal_reg * d_tc_loss

        # Run discriminator optimizer
        self.optimizer_d.zero_grad()
        d_tc_loss.backward()
        self.optimizer_d.step()

        if storer is not None:
            storer['discrim_loss'].append(d_tc_loss.item())

        return vae_loss


class BtcvaeLoss(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling

    def __call__(self, data, recon_batch, latent_dist, is_train, storer,
                 latent_sample=None):
        storer = self._pre_call(is_train, storer)
        batch_size, latent_dim = latent_sample.shape

        rec_loss = _reconstruction_loss(data, recon_batch,
                                        storer=storer,
                                        distribution=self.rec_dist)
        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)

        # total loss
        loss = rec_loss + (self.alpha * mi_loss +
                           self.beta * tc_loss +
                           anneal_reg * self.gamma * dw_kl_loss)

        if storer is not None:
            storer['loss'].append(loss.item())
            storer['mi_loss'].append(mi_loss.item())
            storer['tc_loss'].append(tc_loss.item())
            storer['dw_kl_loss'].append(dw_kl_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss

def DIP(mu_dist,lambda_od,lambda_d):
    #mu [B,D]
        mu=mu_dist[0]
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        # Add Variance for DIP Loss II
        #cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0) # [D x D]
        # For DIp Loss I
        cov_z = cov_mu

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = lambda_od * torch.sum(cov_offdiag ** 2) + lambda_d * torch.sum((cov_diag - 1) ** 2)
        return dip_loss
        
def DIP_group(dist_z, dist_w, lambda_):
        #log_var_z=dist_z[1]
        #log_var_w=dist_w[1]
        mu_z=dist_z[0]
        mu_w=dist_z[0]
        centered_mu_z = mu_z - mu_z.mean(dim=1, keepdim = True) # [B x D1]
        centered_mu_w = mu_w - mu_w.mean(dim=1, keepdim = True) # [B x D2]
        cov_mu_zw = centered_mu_z.t().matmul(centered_mu_w).squeeze() # [D1 X D2]
        cov_mu_wz = centered_mu_w.t().matmul(centered_mu_z).squeeze() # [D2 X D1]

        # Add Variance for DIP Loss II
        #cov_zw = cov_mu_zw + torch.mean(torch.diagonal((2. * log_var_z).exp(), dim1 = 0), dim = 0) # [D1 x D2]
        #cov_wz = cov_mu_wz + torch.mean(torch.diagonal((2. * log_var_w).exp(), dim1 = 0), dim = 0) # [D2 x D1]

        # For DIp Loss I
        cov_zw = cov_mu_zw
        cov_wz = cov_mu_wz

        dip_loss = lambda_ * torch.sum(cov_zw ** 2) + lambda_ * torch.sum(cov_wz  ** 2)
        return dip_loss       
        
def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
    """
    Calculates the per image reconstruction loss for a batch of data. I.e. negative
    log likelihood.

    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Shape : (batch_size, n_chan,
        height, width).

    recon_data : torch.Tensor
        Reconstructed data. Shape : (batch_size, n_chan, height, width).

    distribution : {"bernoulli", "gaussian", "laplace"}
        Distribution of the likelihood on the each pixel. Implicitely defines the
        loss Bernoulli corresponds to a binary cross entropy (bse) loss and is the
        most commonly used. It has the issue that it doesn't penalize the same
        way (0.1,0.2) and (0.4,0.5), which might not be optimal. Gaussian
        distribution corresponds to MSE, and is sometimes used, but hard to train
        ecause it ends up focusing only a few pixels that are very wrong. Laplace
        distribution corresponds to L1 solves partially the issue of MSE.

    storer : dict
        Dictionary in which to store important variables for vizualisation.

    Returns
    -------
    loss : torch.Tensor
        Per image cross entropy (i.e. normalized per batch but not pixel and
        channel)
    """
    batch_size, n_chan, height, width = recon_data.size()
    is_colored = n_chan == 3

    if distribution == "bernoulli":
        loss = F.binary_cross_entropy(recon_data, data, reduction="sum")
    elif distribution == "gaussian":
        # loss in [0,255] space but normalized by 255 to not be too big
        loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255
    elif distribution == "laplace":
        # loss in [0,255] space but normalized by 255 to not be too big but
        # multiply by 255 and divide 255, is the same as not doing anything for L1
        loss = F.l1_loss(recon_data, data, reduction="sum")
        loss = loss * 3  # emperical value to give similar values than bernoulli => use same hyperparam
        loss = loss * (loss != 0)  # masking to avoid nan
    else:
        assert distribution not in RECON_DIST
        raise ValueError("Unkown distribution: {}".format(distribution))

    loss = loss / batch_size

    if storer is not None:
        storer['recon_loss'].append(loss.item())

    return loss


def _kl_normal_loss(mean, logvar, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl

def _kl_normal_loss_w(mean, logvar, label, storer=None):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    storer : dict
        Dictionary in which to store important variables for vizualisation.
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + (mean-label.float()).pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    if storer is not None:
        storer['kl_loss'].append(total_kl.item())
        for i in range(latent_dim):
            storer['kl_loss_' + str(i)].append(latent_kl[i].item())

    return total_kl



def _permute_dims(latent_sample):
    """
    Implementation of Algorithm 1 in ref [1]. Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).

    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).

    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).

    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

# Batch group-TC specific
# TO-DO: test if mss is better!
def _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z,latent_sample_w, latent_dist_z, latent_dist_w,n_data, is_mss=True):
    batch_size, hidden_dim_z = latent_sample_z.shape
    batch_size, hidden_dim_w = latent_sample_w.shape
    hidden_dim=hidden_dim_z+hidden_dim_w
    latent_dist=(torch.cat([latent_dist_z[0],latent_dist_w[0]],dim=-1), torch.cat([latent_dist_z[1],latent_dist_w[1]],dim=-1))
    latent_sample=torch.cat([latent_sample_z,latent_sample_w],dim=-1)
    
    # calculate log q(z,w|x)
    log_q_zwCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z,w)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pzw = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qzqw = matrix_log_density_gaussian(latent_sample, *latent_dist)
    mat_log_qz = matrix_log_density_gaussian(latent_sample_z, *latent_dist_z)
    mat_log_qw = matrix_log_density_gaussian(latent_sample_w, *latent_dist_w)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qzqw = mat_log_qzqw + log_iw_mat.view(batch_size, batch_size, 1)
        log_iw_mat_z = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_z.device)
        mat_log_qz = mat_log_qz + log_iw_mat_z.view(batch_size, batch_size, 1)        
        log_iw_mat_w = log_importance_weight_matrix(batch_size, n_data).to(latent_sample_w.device)
        mat_log_qw = mat_log_qw + log_iw_mat_w.view(batch_size, batch_size, 1)

    log_qzw = torch.logsumexp(mat_log_qzqw.sum(2), dim=1, keepdim=False)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_qw = torch.logsumexp(mat_log_qw.sum(2), dim=1, keepdim=False)
    log_prod_qzqw = log_qz + log_qw

    return log_pzw, log_qzw, log_prod_qzqw, log_q_zwCx


class BtcvaeLoss_property(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=2., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling
        self.mse_loss=torch.nn.MSELoss(reduction="sum")

    def __call__(self, data, label, recon_batch, latent_dist_z, latent_dist_w, is_train, storer,
                 latent_sample_z=None,latent_sample_w=None,p_pred=None):
        
        storer = self._pre_call(is_train, storer)
        latent_sample=torch.cat([latent_sample_w,latent_sample_z],dim=-1)
        batch_size, latent_dim = latent_sample.shape
        latent_dist=(torch.cat([latent_dist_w[0], latent_dist_z[0]],dim=-1),torch.cat([latent_dist_w[1], latent_dist_z[1]],dim=-1))
        num_prop=len(label[0])

        #reconstruction error
        rec_loss = _reconstruction_loss(data, recon_batch[0],
                                        storer=storer,
                                        distribution=self.rec_dist)
        rec_loss_prop=[]
        for idx in range(num_prop):
                rec_loss_prop.append(self.mse_loss(recon_batch[1][:,idx],label[:,idx].float()))

        rec_loss_prop_all= sum(rec_loss_prop)
        #kl loss for z seperately
        kl_loss = _kl_normal_loss(*latent_dist, storer) 
        
        #mse loss of p(y|X)
        #property_prediction_loss=self.mse_loss(p_pred,label.float())
        
        
        #total correlation loss of all latents for pairwise disentangelment for mutiple properties
        log_pw, log_qw, log_prod_qwi, log_q_wCx = _get_log_pz_qz_prodzi_qzCx(latent_sample_w,
                                                                             latent_dist_w,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_wCx - log_qw).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qw - log_prod_qwi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qwi - log_pw).mean()
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)                
        #pairwise_tc_loss=self.alpha * mi_loss + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss
        pairwise_tc_loss=self.beta * tc_loss 
        #total correlation loss between w and z (groupwise disentangelment)
        log_pwz, log_qwz, log_prod_qwqz, log_q_wzCx = _get_log_pzw_qzw_prodzw_qzwCx(latent_sample_z,
                                                                             latent_sample_w,  
                                                                             latent_dist_z,
                                                                             latent_dist_w,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)        
        #TC[z,w] = KL[q(z,w)||\z,w]
        groupwise_tc_loss =self.beta * (log_qwz - log_prod_qwqz).mean()        
        


         
        # total loss
        loss = rec_loss +pairwise_tc_loss+200*rec_loss_prop_all + groupwise_tc_loss + kl_loss
        if storer is not None:
            storer['loss'].append(loss.item())
            #storer['mi_loss'].append(mi_loss.item())
            #storer['tc_loss'].append(tc_loss.item())
            #storer['dw_kl_loss'].append(dw_kl_loss.item())
            #storer['property_prediction_loss'].append(property_prediction_loss.item())
            storer['groupwise_tc_loss'].append(groupwise_tc_loss.item())
            storer['pairwise_tc_loss'].append(pairwise_tc_loss.item())
            for idx in range(num_prop):
              storer['rec_property_loss'+str(idx)].append(rec_loss_prop[idx].item())
#            storer['rec_property_loss2'].append(rec_loss_prop2.item())
#            storer['rec_property_loss3'].append(rec_loss_prop3.item())
#            storer['rec_property_loss4'].append(rec_loss_prop4.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss

class BtcvaeLoss_property_tc(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=2., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling
        self.mse_loss=torch.nn.MSELoss(reduction="mean")

    def __call__(self, data, label, recon_batch, latent_dist_z, latent_dist_w, is_train, storer,
                 latent_sample_z=None,latent_sample_w=None,p_pred=None):
        
        storer = self._pre_call(is_train, storer)
        latent_sample=torch.cat([latent_sample_w,latent_sample_z],dim=-1)
        batch_size, latent_dim = latent_sample.shape
        latent_dist=(torch.cat([latent_dist_w[0], latent_dist_z[0]],dim=-1),torch.cat([latent_dist_w[1], latent_dist_z[1]],dim=-1))
        num_prop=len(label[0])

        #reconstruction error
        rec_loss = _reconstruction_loss(data, recon_batch[0],
                                        storer=storer,
                                        distribution=self.rec_dist)
        rec_loss_prop=[]
        for idx in range(num_prop):
                rec_loss_prop.append(self.mse_loss(recon_batch[1][:,idx],label[:,idx].float()))

        rec_loss_prop_all= sum(rec_loss_prop)
        
        #mse loss of p(y|X)
        #property_prediction_loss=self.mse_loss(p_pred,label.float())
        
        
        #total correlation loss of all latents for pairwise disentangelment for mutiple properties
        log_pw, log_qw, log_prod_qwi, log_q_wCx = _get_log_pz_qz_prodzi_qzCx(latent_sample,
                                                                             latent_dist,
                                                                             self.n_data,
                                                                             is_mss=self.is_mss)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_wCx - log_qw).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qw - log_prod_qwi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qwi - log_pw).mean()
        anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal)
                      if is_train else 1)                
        pairwise_tc_loss=self.alpha * mi_loss + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss
 

         
        # total loss
        loss = rec_loss +pairwise_tc_loss+200*rec_loss_prop_all
        if storer is not None:
            storer['loss'].append(loss.item())
            #storer['mi_loss'].append(mi_loss.item())
            #storer['tc_loss'].append(tc_loss.item())
            #storer['dw_kl_loss'].append(dw_kl_loss.item())
            #storer['property_prediction_loss'].append(property_prediction_loss.item())
            #storer['groupwise_tc_loss'].append(groupwise_tc_loss.item())
            storer['pairwise_tc_loss'].append(pairwise_tc_loss.item())
            for idx in range(num_prop):
              storer['rec_property_loss'+str(idx)].append(rec_loss_prop[idx].item())
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss
    
class SemivaeLoss_property(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling
        self.mse_loss=torch.nn.MSELoss()

    def __call__(self, data, label, recon_batch, latent_dist_z, latent_dist_w, is_train, storer,
                 latent_sample_z=None,latent_sample_w=None,p_pred=None):
        
        storer = self._pre_call(is_train, storer)
        latent_sample=torch.cat([latent_sample_w,latent_sample_z],dim=-1)
        batch_size, latent_dim = latent_sample.shape
        latent_dist=(torch.cat([latent_dist_w[0], latent_dist_z[0]],dim=-1),torch.cat([latent_dist_w[1], latent_dist_z[1]],dim=-1))
        
        num_prop=len(label[0])

        #reconstruction error
        rec_loss = _reconstruction_loss(data, recon_batch[0],
                                        storer=storer,
                                        distribution=self.rec_dist)
        rec_loss_prop=[]
        for idx in range(num_prop):
                rec_loss_prop.append(self.mse_loss(recon_batch[1][:,idx],label[:,idx].float()))

        rec_loss_prop_all= sum(rec_loss_prop)  
        #kl loss for z seperately
        kl_loss = _kl_normal_loss(*latent_dist, storer) 
        

         
        # total loss
        loss = rec_loss +200*rec_loss_prop_all + kl_loss
        if storer is not None:
            storer['loss'].append(loss.item())
            #storer['mi_loss'].append(mi_loss.item())
            #storer['tc_loss'].append(tc_loss.item())
            #storer['dw_kl_loss'].append(dw_kl_loss.item())
            #storer['property_prediction_loss'].append(property_prediction_loss.item())
            #storer['groupwise_tc_loss'].append(groupwise_tc_loss.item())
           # storer['pairwise_tc_loss'].append(pairwise_tc_loss.item())
            for idx in range(num_prop):
              storer['rec_property_loss'+str(idx)].append(rec_loss_prop[idx].item())
              
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss    
    
class CsvaeLoss_property(BaseLoss):
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]

    Parameters
    ----------
    n_data: int
        Number of data in the training set

    alpha : float
        Weight of the mutual information term.

    beta : float
        Weight of the total correlation term.

    gamma : float
        Weight of the dimension-wise KL term.

    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """

    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
        super().__init__(**kwargs)
        self.n_data = n_data
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss  # minibatch stratified sampling
        self.mse_loss=torch.nn.MSELoss()

    def __call__(self, data, label, recon_batch, latent_dist_z, latent_dist_w, is_train, storer,
                 latent_sample_z=None,latent_sample_w=None,p_pred=None):
        
        storer = self._pre_call(is_train, storer)
        latent_sample=torch.cat([latent_sample_w,latent_sample_z],dim=-1)
        batch_size, latent_dim = latent_sample.shape
        latent_dist=(torch.cat([latent_dist_w[0], latent_dist_z[0]],dim=-1),torch.cat([latent_dist_w[1], latent_dist_z[1]],dim=-1))
        num_prop=len(label[0])
        #reconstruction error
        rec_loss = _reconstruction_loss(data, recon_batch[0],
                                        storer=storer,
                                        distribution=self.rec_dist)
        

        rec_loss_prop=[]
        for idx in range(num_prop):
                rec_loss_prop.append(self.mse_loss(recon_batch[1][:,idx],label[:,idx].float()))

        rec_loss_prop_all= sum(rec_loss_prop)
        #kl loss for z seperately
        kl_loss = _kl_normal_loss(*latent_dist_z, storer) 
        kl_loss_w = _kl_normal_loss_w(*latent_dist_w,label, storer) 
        

         
        # total loss
        loss1 = rec_loss -1000*rec_loss_prop_all + kl_loss + kl_loss_w
        loss2= 100*rec_loss_prop_all
        if storer is not None:
            storer['loss1'].append(loss1.item())
            storer['loss2'].append(loss2.item())
            #storer['tc_loss'].append(tc_loss.item())
            #storer['dw_kl_loss'].append(dw_kl_loss.item())
            #storer['property_prediction_loss'].append(property_prediction_loss.item())
            #storer['groupwise_tc_loss'].append(groupwise_tc_loss.item())
           # storer['pairwise_tc_loss'].append(pairwise_tc_loss.item())
            # computing this for storing and comparaison purposes
            _ = _kl_normal_loss(*latent_dist, storer)

        return loss1,loss2        
    
    
#class DIPvaeLoss_property(BaseLoss):
#    """
#    Compute the decomposed KL loss with either minibatch weighted sampling or
#    minibatch stratified sampling according to [1]
#
#    Parameters
#    ----------
#    n_data: int
#        Number of data in the training set
#
#    alpha : float
#        Weight of the mutual information term.
#
#    beta : float
#        Weight of the total correlation term.
#
#    gamma : float
#        Weight of the dimension-wise KL term.
#
#    is_mss : bool
#        Whether to use minibatch stratified sampling instead of minibatch
#        weighted sampling.
#
#    kwargs:
#        Additional arguments for `BaseLoss`, e.g. rec_dist`.
#
#    References
#    ----------
#       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
#       autoencoders." Advances in Neural Information Processing Systems. 2018.
#    """
#
#    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):
#        super().__init__(**kwargs)
#        self.n_data = n_data
#         # minibatch stratified sampling
#        self.mse_loss=torch.nn.MSELoss()
#
#    def __call__(self, data, label, recon_batch, latent_dist_z, latent_dist_w, is_train, storer,
#                 latent_sample_z=None,latent_sample_w=None,p_pred=None):
#        
#        storer = self._pre_call(is_train, storer)
#        latent_sample=torch.cat([latent_sample_z,latent_sample_w],dim=-1)
#        batch_size, latent_dim = latent_sample.shape
#
#        
#        #reconstruction error
#        rec_loss = _reconstruction_loss(data, recon_batch[0],
#                                        storer=storer,
#                                        distribution=self.rec_dist)
#        
#        rec_loss_prop=self.mse_loss(recon_batch[1],label.float())
#        
#        
#        #kl loss for z seperately
#        kl_loss = _kl_normal_loss(*latent_dist_z, storer)+ _kl_normal_loss(*latent_dist_w, storer)   
#
#        #mse loss of p(y|X)
#        property_prediction_loss=self.mse_loss(p_pred,label.float())
#        
#        
#        #dip loss of w for pairwise disentangelment for mutiple properties
#        if latent_dist_w[0].size(1) > 1:
#          pairwise_loss= DIP(latent_dist_w, 10, 1)
#        else:
#           pairwise_loss=torch.zeros(1) 
#
#        #dip loss between w and z (groupwise disentangelment)
#        groupwise_loss = DIP_group(latent_dist_z, latent_dist_w, 50)        
#        
#
#        # total loss
#        if latent_dist_w[0].size(1) > 1:        
#          loss = rec_loss + rec_loss_prop + property_prediction_loss + kl_loss+pairwise_loss+groupwise_loss
#        else:
#           loss = rec_loss  + 5*kl_loss #50*rec_loss_prop+ 0*property_prediction_loss +groupwise_loss
#    
#            
#        if storer is not None:
#            storer['loss'].append(loss.item())
#            #storer['property_prediction_loss'].append(property_prediction_loss.item())
#            #storer['groupwise_loss'].append(groupwise_loss.item())
#            #storer['pairwise_loss'].append(pairwise_loss.item())
#            storer['rec_property_loss'].append(rec_loss_prop.item())
#            # computing this for storing and comparaison purposes
#            _ = _kl_normal_loss(*latent_dist_z, storer)
#
#        return loss    