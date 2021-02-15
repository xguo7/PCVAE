"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from disvae.utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder

MODELS = ["Burgess","ControlVAE",'SemiVAE','CsVAE']


def init_specific_model(model_type, img_size, latent_dim, num_prop=None):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    if model_type =='Burgess':
        model_type = model_type.lower().capitalize()
        if model_type not in MODELS:
            err = "Unkown model_type={}. Possible values: {}"
            raise ValueError(err.format(model_type, MODELS))
    
        encoder = get_encoder(model_type)
        decoder = get_decoder(model_type)
        model = VAE(img_size, encoder, decoder, latent_dim)
        model.model_type = model_type  # store to help reloading
        
    elif model_type == "ControlVAE": 
        #model_type= model_type.lower().capitalize()
        if model_type not in MODELS:
                err = "Unkown model_type={}. Possible values: {}"
                raise ValueError(err.format(model_type, MODELS))       
        
        encoder = get_encoder(model_type)
        decoder = get_decoder(model_type)
        model = ControlVAE(img_size, encoder, decoder, latent_dim, num_prop)
        model.model_type = model_type  # store to help reloading        

    elif model_type == "SemiVAE": 
        #model_type= model_type.lower().capitalize()
        if model_type not in MODELS:
                err = "Unkown model_type={}. Possible values: {}"
                raise ValueError(err.format(model_type, MODELS))       
        
        encoder = get_encoder(model_type)
        decoder = get_decoder(model_type)
        model = SemiVAE(img_size, encoder, decoder, latent_dim, num_prop)
        model.model_type = model_type  # store to help reloading 
        
    elif model_type == "CsVAE": 
        #model_type= model_type.lower().capitalize()
        if model_type not in MODELS:
                err = "Unkown model_type={}. Possible values: {}"
                raise ValueError(err.format(model_type, MODELS))       
        
        encoder = get_encoder(model_type)
        decoderX, decoderY= get_decoder(model_type)
        model = CsVAE(img_size, encoder, decoderX, decoderY,latent_dim, num_prop)
        model.model_type = model_type  # store to help reloading          
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

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

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class ControlVAE_old(nn.Module):
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
            latent_dist_w_mean,latent_dist_w_std,p_pred = self.encoder_w(x,p) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)
        reconstruct,y_reconstruct = self.decoder(latent_sample_z,latent_sample_w)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred

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

class ControlVAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, num_prop):
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
        self.encoder = encoder(img_size, self.latent_dim, self.num_prop)
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

    def forward(self, x, label=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """

        if self.training:
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for training process
        else:            
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)
        
        reconstruct,y_reconstruct = self.decoder(latent_sample_z,latent_sample_w)
        
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred

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
        latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,p)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)        
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)
        
        return latent_sample_z, latent_sample_w
    
    def iterate_get_w(self,label,w_latent_idx, maxIter=20):
        #get the w for a kind of given property
        w_n=label.view(-1,1).to('cuda').float()#[N]
        for iter_index in range(maxIter):      
               summand = self.decoder.property_lin_list[w_latent_idx](w_n)
               w_n1 = label.view(-1,1).to('cuda').float() - summand
               print('Iteration of difference:'+str(torch.abs(w_n-w_n1).mean().item()))
               w_n=w_n1.clone()
        return w_n1.view(-1)    
    
class SemiVAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim, num_prop):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(SemiVAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        self.num_prop=num_prop
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim, self.num_prop)
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

    def forward(self, x, label=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """

        if self.training:
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for training process
        else:            
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)
        
        reconstruct,y_reconstruct = self.decoder(latent_sample_z,latent_sample_w)
        
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred

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
        latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,p)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)        
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)
        
        return latent_sample_z, latent_sample_w
    
    def iterate_get_w(self,label,w_latent_idx, maxIter=20):
        #get the w for a kind of given property
        w_n=label.view(-1,1).to('cuda').float()#[N]
        return w_n.view(-1)        
    
    
class CsVAE(nn.Module):
    def __init__(self, img_size, encoder, decoderX, decoderY, latent_dim, num_prop):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(CsVAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
        
        self.num_prop=num_prop
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim, self.num_prop)
        self.decoderX = decoderX(img_size, self.latent_dim, self.num_prop)
        self.decoderY = decoderY(img_size, self.latent_dim, self.num_prop)

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

    def forward(self, x, label=None):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """

        if self.training:
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for training process
        else:            
            latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,label) #for testing process
        
        latent_sample_z = self.reparameterize(latent_dist_z_mean,latent_dist_z_std)
        latent_sample_w = self.reparameterize(latent_dist_w_mean,latent_dist_w_std)
        
        reconstruct = self.decoderX(latent_sample_z,latent_sample_w)
        y_reconstruct = self.decoderY(latent_sample_z)
        
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)
        return (reconstruct,y_reconstruct), latent_dist_z, latent_dist_w, latent_sample_z,latent_sample_w,p_pred

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
        latent_dist_z_mean,latent_dist_w_mean,latent_dist_z_std,latent_dist_w_std,p_pred = self.encoder(x,p)
        latent_dist_z=(latent_dist_z_mean,latent_dist_z_std)
        latent_dist_w=(latent_dist_w_mean,latent_dist_w_std)        
        latent_sample_z = self.reparameterize(*latent_dist_z)
        latent_sample_w = self.reparameterize(*latent_dist_w)
        
        return latent_sample_z, latent_sample_w
         