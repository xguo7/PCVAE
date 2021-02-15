"""
Module containing the decoders.
"""
import numpy as np

import torch
from torch import nn
from .spectral_norm_fc import spectral_norm_fc

# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    if model_type=='Csvae':
        return eval("Decoder{}X".format(model_type)),eval("Decoder{}Y".format(model_type))
    else:
        return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x

class DecoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10, num_prop=2):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderControlvae, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.num_prop=num_prop
        latent_dim=latent_dim_z+num_prop
        self.sigmoid=torch.nn.Sigmoid()
        
        
        # decoder for the property 
        self.property_lin_list=nn.ModuleList()
        for idx in range(num_prop):
            layers=[]
            layers.append(spectral_norm_fc(nn.Linear(1, hidden_dim_prop).to('cuda')))
            layers.append(nn.ReLU())
            layers.append(spectral_norm_fc(nn.Linear(hidden_dim_prop, 1).to('cuda')))
            if num_prop-idx==4:#if deaing with proprty 0-2pi
               layers.append(nn.ReLU())
            else:
               layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))
            
            
        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z,w):
        batch_size = z.size(0)
        wz=torch.cat([w,z],dim=-1)
        prop=[]
        
        #fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            w_=w[:,idx].view(-1,1)
            prop.append(self.property_lin_list[idx](w_)+w_)
                
        
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(wz))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x, torch.cat(prop,dim=-1)
    

        
class DecoderSemivae(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10, num_prop=2):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderSemivae, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.num_prop=num_prop
        latent_dim=latent_dim_z+num_prop
        self.sigmoid=torch.nn.Sigmoid()
        
    
            
            
        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z,w):
        batch_size = z.size(0)
        wz=torch.cat([w,z],dim=-1)
        prop=[]
        
        #fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            w_=w[:,idx].view(-1,1)
            prop.append(w_)
                
        
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(wz))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x, torch.cat(prop,dim=-1)
    
    
# class DecoderCsvae(nn.Module):
#     def __init__(self, img_size,
#                  latent_dim_z=10, num_prop=2):
#         r"""Decoder of the model proposed in [1].

#         Parameters
#         ----------
#         img_size : tuple of ints
#             Size of images. E.g. (1, 32, 32) or (3, 64, 64).

#         latent_dim : int
#             Dimensionality of latent output.

#         Model Architecture (transposed for decoder)
#         ------------
#         - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
#         - 2 fully connected layers (each of 256 units)
#         - Latent distribution:
#             - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

#         References:
#             [1] Burgess, Christopher P., et al. "Understanding disentangling in
#             $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
#         """
#         super(DecoderCsvae, self).__init__()

#         self.decoderX=DecoderX(img_size,latent_dim_z, num_prop)
#         self.decoderY=DecoderY(img_size,latent_dim_z, num_prop)
#         self.decoderX_parameters=self.decoderX.parameters()
#         self.decoderY_parameters=self.decoderY.parameters()
#     def forward(self, z,w):
#         return self.decoderX(z,w), self.decoderY(z)
    
class DecoderCsvaeX(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10, num_prop=2):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderCsvaeX, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size
        self.num_prop=num_prop
        latent_dim=latent_dim_z+num_prop
            
            
        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z,w):
        batch_size = z.size(0)
        wz=torch.cat([w,z],dim=-1)
                
        
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(wz))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x   
    
class DecoderCsvaeY(nn.Module):
    def __init__(self, img_size,
                 latent_dim_z=10, num_prop=2):
        r"""Decoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(DecoderCsvaeY, self).__init__()

        # Layer parameters
        hidden_dim_prop=50
        self.img_size = img_size
        # Shape required to start transpose convs

        self.img_size = img_size
        self.num_prop=num_prop
        self.sigmoid=torch.nn.Sigmoid()
        
        
        # decoder for the property 
        self.property_lin_list=nn.ModuleList()
        for idx in range(num_prop):
            layers=[]
            layers.append(nn.Linear(latent_dim_z, hidden_dim_prop).to('cuda'))
            layers.append(nn.ReLU())
            layers.append(spectral_norm_fc(nn.Linear(hidden_dim_prop, 1).to('cuda')))
            if num_prop-idx==4:#if deaing with proprty 0-2pi
               layers.append(nn.ReLU())
            else:
               layers.append(nn.Sigmoid())
            self.property_lin_list.append(nn.Sequential(*layers))

    def forward(self, z):
        prop=[]        
        #fully connected process for reconstruct the properties
        for idx in range(self.num_prop):
            #z_=z.view(-1,1)
            prop.append(self.property_lin_list[idx](z))
                

        return torch.cat(prop,dim=-1)    