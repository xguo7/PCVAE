# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:12:06 2020

@author: gxjco
"""
import numpy as np

mig=np.array([[0.0006,0.0006,0.0006],[0.2339,0.008,0.0131],[0.02,0.0428,0.0319],[0.001,0.0015,0.0012]])

def draw_mig(mig):
       import matplotlib.pyplot as plt
       import seaborn as sns
       sns.set()
       
       #import pandas as pd 
       #df = pd.DataFrame(mig)

       f, ax = plt.subplots(figsize=(4, 3))

       sns.heatmap(mig, annot=True, linewidths=1, ax=ax)

       label_y = ax.get_yticklabels()
       plt.setp(label_y, rotation=360, horizontalalignment='right')
       label_x = ax.get_xticklabels()
       plt.setp(label_x, rotation=45, horizontalalignment='right')
       #sns_plot.savefig(os.path.join(self.save_dir, 'avgMI'))
      
draw_mig(mig)

I_mat=np.diag((1,1))
MI_mat=mig[1:]
print(np.linalg.norm(I_mat - MI_mat,ord=2))