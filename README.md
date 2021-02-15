#for training the model on dsprites datasets:
%run main.py control_dsprites -d dsprites --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

#for testing the model on dsprites datasets:
%run main_viz.py control_dsprites all -e 60 -d dsprites


#for training the model on 3dshapes datasets:
%run main.py control_3dshapes -d 3dshapes --lr 0.0005 -e 61 -b 64 -l btcvae_property -num_prop 3

#for testing the model on 3dshapes datasets:
%run main_viz.py control_3dshapes all -e 60 -d 3dshapes

#All the parameters can be modofied in the file: hyperparam.ini
#All the generated results will be stored in the folder: results
 
