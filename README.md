# Property Controllable Variational Autoencoder via Invertible Mutual Dependence.
![image_text](images/NEC-DGT.png)
This repository is the official Tensorflow implementation of PCVAE, a property controllable variational autoencoder.

The relevant paper is ["Property Controllable Variational Autoencoder"](http://mason.gmu.edu/~lzhao9/materials/papers/ICDM_2019_NEC_DGT-final.pdf).

[Xiaojie Guo](https://sites.google.com/view/xiaojie-guo-personal-site), [Yuanqi Du](https://yuanqidu.github.io/), [Liang Zhao](http://mason.gmu.edu/~lzhao9/).

## Installation
Install Tensorflow following the instuctions on the official website. The code has been tested over Tensorflow 1.13.1 version.

## Datasets

The dsprite dataset can be found and downloaded at [Datasets for dSprite](https://github.com/deepmind/dsprites-dataset). 

The 3Dshape dataset can be found and downloaded at [Datasets for 3Dshape](https://github.com/deepmind/3d-shapes). 


## Run the code


# for training the model on dsprites datasets:
%run main.py control_dsprites -d dsprites --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

# for testing the model on dsprites datasets:
%run main_viz.py control_dsprites all -e 60 -d dsprites


# for training the model on 3dshapes datasets:
%run main.py control_3dshapes -d 3dshapes --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

# for testing the model on 3dshapes datasets:
%run main_viz.py control_3dshapes all -e 60 -d 3dshapes

# All the parameters can be modofied in the file: hyperparam.ini
# All the generated results will be stored in the folder: results
 
