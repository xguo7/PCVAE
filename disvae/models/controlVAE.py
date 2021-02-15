"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder

MODELS = ["ControlVAE"]


def init_specific_model(model_type, img_size, latent_dim, num_prop):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    if isinstance(model_type,list) is list:
        model_type_z = model_type[0].lower().capitalize()
        model_type_w = model_type[1].lower().capitalize()
        if model_type_z not in MODELS or model_type_w not in MODELS:
            err = "Unkown model_type={}. Possible values: {}"
            raise ValueError(err.format(model_type, MODELS))
    else:
            raise ValueError("need two model_type")        

    encoder_z = get_encoder(model_type[0])
    encoder_w = get_encoder(model_type[1])
    decoder = get_decoder(model_type[1])
    model = ControlVAE(img_size, encoder_z, encoder_w, decoder, latent_dim, num_prop)
    model.model_type = model_type  # store to help reloading
    return model


class ControlVAE(nn.Module):
    def __init__(self, img_size, encoder_z, encoder_w, decoder, latent_dim, num_prop):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(ControlVAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        self.num_prop=num_prop
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder_z = encoder_z(img_size, self.latent_dim)
        self.encoder_w = encoder_w(img_size, self.latent_dim, self.num_prop)
        self.decoder = decoder(img_size, self.latent_dim, self.num_prop)
        

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x,p=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        
        latent_dist_z_mean,latent_dist_z_std = self.encoder_z(x)
        if self.training:
            latent_dist_w_mean,latent_dist_w_std,p_pred = self.encoder_w(x,p) #for training process
        else:            
            latent_dist_w_mean,latent_dist_w_std = self.encoder_w(x,p) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)
        reconstruct = self.decoder(latent_sample_z,latent_sample_w)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return reconstruct, latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x,p=None):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist_z = self.encoder_z(x)
        latent_dist_w = self.encoder_w(x,p)
        
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)
        
        return latent_sample_z, latent_sample_w

