# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:09:45 2020

@author: xguo7
"""

#for training the model on dsprites datasets:
%run main.py control_dsprites_multiple3 -d dsprites --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

#for testing the model on dsprites datasets:
%run main_viz.py control_dsprites all -e 60 -d dsprites


#for training the model on 3dshapes datasets:
%run main.py control_3dshapes_multiple3 -d 3dshapes --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

#for testing the model on 3dshapes datasets:
%run main_viz.py control_3dshapes_multiple3 all -e 40 -d 3dshapes