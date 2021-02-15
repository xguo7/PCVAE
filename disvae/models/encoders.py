"""
Module containing the encoders.
"""
import numpy as np

import torch
from torch import nn
import torch.functional as F

# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256

        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class EncoderControlvae_old(nn.Module):
    def __init__(self, img_size,
                 latent_dim=1,num_prop=2,if_given_property=False):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderControlvae, self).__init__()
        self.if_given_property=if_given_property
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=20
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.num_prop=num_prop
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1_list = []
        self.lin2_list = []
        self.prop_lin3_list=[]
        # Fully connected layers for mean and variance
        self.prop_mu_logvar_list=[]  
        self.sigmoid=torch.nn.Sigmoid().to('cuda')
        for idx in range(num_prop):
           self.lin1_list.append(nn.Linear(np.product(self.reshape), hidden_dim).to('cuda')) 
           self.lin2_list.append(nn.Linear(hidden_dim, 1).to('cuda')) 
           self.prop_lin3_list.append(nn.Linear(1,hidden_dim_prop).to('cuda'))       
           self.prop_mu_logvar_list.append(nn.Linear(hidden_dim_prop, 2).to('cuda'))

    def forward(self, x, prop=None):
        batch_size = x.size(0)
        if self.if_given_property is False:
            # Convolutional layers with ReLu activations
            prop=torch.zeros(batch_size,self.num_prop).to('cuda')
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            if self.img_size[1] == self.img_size[2] == 64:
                x = torch.relu(self.conv_64(x))
            for idx in range(self.num_prop): 
                    # Fully connected layers with ReLu activations
                    x_= torch.relu(self.lin1_list[idx](x.view((batch_size, -1))))            
                    prop[:,idx] = self.sigmoid(self.lin2_list[idx](x_).view(-1)).to('cuda')  #get the property  #batch*num_prop
        else:
              prop=prop

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        w=[]
        
        for idx in range(self.num_prop): 
            prop_=prop[:,idx].view(-1,1).to('cuda')
            w.append(self.prop_lin3_list[idx](prop_))
        mu=[]
        logvar=[]
        for idx in range(self.num_prop): 
           mu_logvar = self.prop_mu_logvar_list[idx](w[idx])
           mu_, logvar_ = mu_logvar.view(-1, 1, 2).unbind(-1)
           mu.append(mu_)
           logvar.append(logvar_)
           
        if self.if_given_property is False:
           return torch.cat(mu,dim=-1), torch.cat(logvar,dim=-1), prop
        else:
           return torch.cat(mu,dim=-1), torch.cat(logvar,dim=-1)
       
class EncoderControlvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,num_prop=2,if_given_property=False):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderControlvae, self).__init__()
        self.if_given_property=if_given_property
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.num_prop=num_prop
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)


        # Fully connected layers for property prediction
        self.prop_lin1_list = []
        self.prop_lin2_list = []
        self.prop_lin3_list=[]
        # Fully connected layers for mean and variance
        self.prop_mu_logvar_list=[]  
        self.sigmoid=torch.nn.Sigmoid().to('cuda')
        for idx in range(num_prop):
           self.prop_lin1_list.append(nn.Linear(np.product(self.reshape), hidden_dim_prop).to('cuda')) 
           self.prop_lin2_list.append(nn.Linear(hidden_dim_prop, 1).to('cuda')) 
           self.prop_lin3_list.append(nn.Linear(hidden_dim,hidden_dim_prop).to('cuda'))       
           self.prop_mu_logvar_list.append(nn.Linear(hidden_dim_prop, 2).to('cuda'))
 
          
        # Fully connected layers for unobversed properties
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, (self.latent_dim+self.num_prop) * 2)           

    def forward(self, x, label, prop=None):
        
        batch_size = x.size(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))        
        x_z = x.view((batch_size, -1))
        x_z = torch.relu(self.lin1(x_z))
        x_z = torch.relu(self.lin2(x_z))        
        
        
        #generate the property and latent w        
#        if self.if_given_property is False:
            # Convolutional layers with ReLu activations
#            prop=torch.zeros(batch_size,self.num_prop).to('cuda')
#            for idx in range(self.num_prop): 
#                    # Fully connected layers with ReLu activations
#                    x_= torch.relu(self.prop_lin1_list[idx](x_z))            
#                    prop[:,idx] = self.sigmoid(self.prop_lin2_list[idx](x_).view(-1)).to('cuda')  #get the property  #batch*num_prop
#        else:
#              prop=prop
#        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
#        w=[]       
#        for idx in range(self.num_prop): 
#            #prop_= label[:,idx].view(-1,1).to('cuda').float()
#            prop_= x_z.to('cuda').float()
#            w.append(torch.relu(self.prop_lin3_list[idx](prop_)))
#        mu_w=[]
#        logvar_w=[]
#        for idx in range(self.num_prop): 
#           mu_logvar = self.prop_mu_logvar_list[idx](w[idx])
#           mu_, logvar_ = mu_logvar.view(-1, 1, 2).unbind(-1)
#           mu_w.append(mu_)
#           logvar_w.append(logvar_)
           
           
        #generate the latent z:        

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x_z)
        mu, logvar = mu_logvar.view(-1, self.latent_dim+self.num_prop, 2).unbind(-1)        
           
        return mu[:,:self.latent_dim],mu[:,self.latent_dim:], logvar[:,:self.latent_dim], logvar[:,self.latent_dim:], prop

       
class EncoderSemivae(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,num_prop=2,if_given_property=False):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderSemivae, self).__init__()
        self.if_given_property=if_given_property
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.latent_dim = latent_dim+num_prop
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.num_prop=num_prop
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

 
          
        # Fully connected layers for unobversed properties
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)           

    def forward(self, x, label, prop=None):
        
        batch_size = x.size(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))        
        x_z = x.view((batch_size, -1))
        x_z = torch.relu(self.lin1(x_z))
        x_z = torch.relu(self.lin2(x_z))        
        
        
        mu_logvar = self.mu_logvar_gen(x_z)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        mu_z=mu[:,:-self.num_prop]
        logvar_z=logvar[:,:-self.num_prop]
        mu_w=mu[:,-self.num_prop:]
        logvar_w=logvar[:,-self.num_prop:]
        
           
        if self.if_given_property is False:
           return mu_z,mu_w, logvar_z, logvar_w, prop
        else:
           return mu_z,mu_w, logvar_z, logvar_w, prop

class EncoderCsvae(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10,num_prop=2,if_given_property=False):
        r"""Encoder of the model proposed in [1].

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
        super(EncoderCsvae, self).__init__()
        self.if_given_property=if_given_property
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        hidden_dim_prop=50
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.num_prop=num_prop
        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1z = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2z = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3z = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)


        self.lin1z = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2z = nn.Linear(hidden_dim, hidden_dim)
        # Fully connected layers for mean and variance
        self.mu_logvar_gen_z = nn.Linear(hidden_dim, self.latent_dim * 2)           

        self.conv1w = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2w = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3w = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.lin1w=nn.ModuleList()
        self.lin2w=nn.ModuleList()  
        self.mu_logvar_gen_w=nn.ModuleList()
        for idx in range(self.num_prop):
          self.lin1w.append(nn.Linear(np.product(self.reshape)+1, hidden_dim))
          self.lin2w.append(nn.Linear(hidden_dim, hidden_dim))
        # Fully connected layers for mean and variance
          self.mu_logvar_gen_w.append(nn.Linear(hidden_dim, 2))
        
    def forward(self, x, label, prop=None):
        x_init=x
        
        batch_size = x.size(0)
        x = torch.relu(self.conv1z(x))
        x = torch.relu(self.conv2z(x))
        x = torch.relu(self.conv3z(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))        
        x_z = x.view((batch_size, -1))
        x_z = torch.relu(self.lin1z(x_z))
        x_z = torch.relu(self.lin2z(x_z))                
        
        mu_logvar_z = self.mu_logvar_gen_z(x_z)
        mu_z, logvar_z = mu_logvar_z.view(-1, self.latent_dim, 2).unbind(-1)

        x_ = torch.relu(self.conv1w(x_init))
        x_ = torch.relu(self.conv2w(x_))
        x_ = torch.relu(self.conv3w(x_))
        if self.img_size[1] == self.img_size[2] == 64:
            x_ = torch.relu(self.conv_64(x_))        
        x_w = x_.view((batch_size, -1))
        mu_w=[]
        logvar_w=[]
        for idx in range(self.num_prop):        
           x_w_y = torch.cat([x_w,label[:,idx].float().view(-1,1)],dim=-1)
           x_w_y = torch.relu(self.lin1w[idx](x_w_y))
           x_w_y = torch.relu(self.lin2w[idx](x_w_y))                
           mu_logvar_w = self.mu_logvar_gen_w[idx](x_w_y)
           mu_,logvar_= mu_logvar_w.view(-1, 1, 2).unbind(-1)
           mu_w.append(mu_)
           logvar_w.append(logvar_)         
           
        if self.if_given_property is False:
           return mu_z,torch.cat(mu_w,dim=-1), logvar_z, torch.cat(logvar_w,dim=-1), prop
        else:
           return mu_z,torch.cat(mu_w,dim=-1), logvar_z, torch.cat(logvar_w,dim=-1)         