import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer

from tqdm import trange, tqdm
import numpy as np
import torch
import sklearn
from disvae.models.losses import get_loss_f
from disvae.utils.math import log_density_gaussian
from disvae.utils.modelIO import save_metadata

TEST_LOSSES_FILE = "test_losses.log"
METRICS_FILENAME = "metrics.log"
METRIC_HELPERS_FILE = "metric_helpers.pth"

def avgMI(metric):
    I_mat=np.diag((1,1,1))
    MI_mat=np.concatenate((metric[-3:][:,2].reshape(3,1),metric[-3:][:,-2:]),axis=1)
    return np.linalg.norm(I_mat - MI_mat,ord=2)
    
def scoreReg(predicY,testY):
    MSE=np.sum(np.power((testY.reshape(-1,1) - predicY),2))/len(testY)
    R2=1-MSE/np.var(testY)
    return MSE, R2

def calc_MI(X,Y,bins=10):
   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return 2*MI/(H_X + H_Y)

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

class Evaluator:
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, loss_f, model_type,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True,
                 metric_type='aveMIG'):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info("Testing Device: {}".format(self.device))
        self.model_type=model_type
        self.metric_type = metric_type

    def __call__(self, data_loader, is_metrics=False, is_losses=True):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_metrics: bool, optional
            Whether to compute and store the disentangling metrics.

        is_losses: bool, optional
            Whether to compute and store the test losses.
        """
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        metric, losses = None, None
        if is_metrics:
            self.logger.info('Computing metrics...')
            if self.metric_type == 'aveMIG':
                #control_metrics=self.compute_control_mse(data_loader)
                #print(control_metrics)
                metrics= self.compute_avgMI2(data_loader)                  
            else:
                metrics = self.compute_metrics(data_loader)
            #self.logger.info('Losses: {}'.format(metrics))
            #save_metadata(metrics, self.save_dir, filename=METRICS_FILENAME)
       
        if is_losses:
            self.logger.info('Computing losses...')
            losses = self.compute_losses(data_loader)
            self.logger.info('Losses: {}'.format(losses))
            save_metadata(losses, self.save_dir, filename=TEST_LOSSES_FILE)

        if is_still_training:
            self.model.train()
            
            
        draw_mig(metrics)
        self.logger.info('Finished evaluating after {:.1f} min.'.format((default_timer() - start) / 60))
        print('avgMI:'+str(avgMI(metrics)))
        return metrics

    def compute_losses(self, dataloader):
        """Compute all test losses.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        storer = defaultdict(list)
        for data, label in tqdm(dataloader, leave=False, disable=not self.is_progress_bar):
            data = data.to(self.device)
            if self.model.num_prop==3:
                labels = np.concatenate([label[:,-4:-3],label[:,-2:]],axis=-1)
            elif self.model.num_prop==2:
                labels = label[:,-self.num_prop:]
            elif self.model.num_prop==4:
                labels = np.concatenate([label[:,-3:-2],label[:,-4],label[:,-2:]],axis=-1)                   
            labels = torch.tensor(labels).to(self.device)

            #try:
            if self.model_type == "ControlVAE":                    
                recon_batch, latent_dist_z, latent_dist_w, latent_sample_z, latent_sample_w,p_pred=self.model(data)
                _ = self.loss_f(data, labels, recon_batch, latent_dist_z, latent_dist_w, self.model.training,
                           storer, latent_sample_z=latent_sample_z, latent_sample_w=latent_sample_w,p_pred=p_pred)            
            else:    
                recon_batch, latent_dist_z, latent_dist_w, latent_sample_z, latent_sample_w,p_pred=self.model(data,labels)
                _ = self.loss_f(data, labels, recon_batch, latent_dist_z, latent_dist_w, self.model.training,
                           storer, latent_sample_z=latent_sample_z, latent_sample_w=latent_sample_w,p_pred=p_pred)            
            #except ValueError:
                # for losses that use multiple optimizers (e.g. Factor)
            #    _ = self.loss_f.call_optimize(data, self.model, None, storer)

            losses = {k: sum(v) / len(dataloader) for k, v in storer.items()}
            return losses

    def compute_metrics(self, dataloader):
        """Compute all the metrics.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        try:
            lat_sizes = dataloader.dataset.lat_sizes
            lat_names = dataloader.dataset.lat_names
        except AttributeError:
            raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))

        self.logger.info("Computing the empirical distribution q(z|x).")
        samples_zwCx, params_zwCx = self._compute_q_zCx(dataloader)
        
        if self.model_type == 'ControlVAE': 
           samples_zCx, samples_wCx = samples_zwCx
           params_zCx, params_wCx = params_zwCx            
           len_dataset, latent_dim = samples_zCx.shape
           len_dataset, num_prop = samples_wCx.shape

           self.logger.info("Estimating the marginal entropy.")
           # marginal entropy H(z_j)
           H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)
           H_w = self._estimate_latent_entropies(samples_wCx, params_wCx)
           # conditional entropy H(z|v)
           samples_zCx = samples_zCx.view(*lat_sizes, latent_dim)
           samples_wCx = samples_wCx.view(*lat_sizes, num_prop)
           params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx)
           params_wCx = tuple(p.view(*lat_sizes, num_prop) for p in params_wCx)
          
           H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)
           H_wCv = self._estimate_H_zCv(samples_wCx, params_wCx, lat_sizes, lat_names)
        
           H_z = H_z.cpu()
           H_zCv = H_zCv.cpu()
           H_w = H_w.cpu()
           H_wCv = H_wCv.cpu()    
            # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
           mut_info_z = - H_zCv + H_z
           mut_info_w = - H_wCv + H_w
           sorted_mut_info_z = torch.sort(mut_info_z, dim=1, descending=True)[0].clamp(min=0)
           sorted_mut_info_w = torch.sort(mut_info_w, dim=1, descending=True)[0].clamp(min=0)
        
           metric_helpers_z = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
           metric_helpers_w = {'marginal_entropies': H_w, 'cond_entropies': H_wCv}
                     
           mig_z = self._mutual_information_gap(sorted_mut_info_z, lat_sizes, storer=metric_helpers_z)
           mig_w = self._mutual_information_gap(sorted_mut_info_w, lat_sizes, storer=metric_helpers_w)
                      
           aam_z = self._axis_aligned_metric(sorted_mut_info_z, storer=metric_helpers_z)
           aam_w = self._axis_aligned_metric(sorted_mut_info_w, storer=metric_helpers_w)
        
           metrics = {'MIG': [mig_z.item(),mig_w.item()], 'AAM': [aam_z.item(),aam_w.item()]}
           torch.save([metric_helpers_z,metric_helpers_w], os.path.join(self.save_dir, METRIC_HELPERS_FILE))
           
        else:     
           len_dataset, latent_dim = samples_zCx.shape
           self.logger.info("Estimating the marginal entropy.")
           # marginal entropy H(z_j)
           H_z = self._estimate_latent_entropies(samples_zCx[0], params_zCx[0])
           # conditional entropy H(z|v)
           samples_zCx = samples_zCx[0].view(*lat_sizes, latent_dim)
           params_zCx = tuple(p.view(*lat_sizes, latent_dim) for p in params_zCx[0])
          
           H_zCv = self._estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)
        
           H_z = H_z.cpu()
           H_zCv = H_zCv.cpu()    
            # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
           mut_info_z = - H_zCv + H_z
           sorted_mut_info_z = torch.sort(mut_info_z, dim=1, descending=True)[0].clamp(min=0)
           
           metric_helpers_z = {'marginal_entropies': H_z, 'cond_entropies': H_zCv}
                     
           mig_z = self._mutual_information_gap(sorted_mut_info_z, lat_sizes, storer=metric_helpers_z)
                      
           aam_z = self._axis_aligned_metric(sorted_mut_info_z, storer=metric_helpers_z)
           
           metrics = {'MIG': mig_z.item(), 'AAM': aam_z.item()}
           torch.save(metric_helpers_z, os.path.join(self.save_dir, METRIC_HELPERS_FILE))
                    
           return metrics

    def _mutual_information_gap(self, sorted_mut_info, lat_sizes, storer=None):
        """Compute the mutual information gap as in [1].

        References
        ----------
           [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
           autoencoders." Advances in Neural Information Processing Systems. 2018.
        """
        # difference between the largest and second largest mutual info
        delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]
        # NOTE: currently only works if balanced dataset for every factor of variation
        # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
        H_v = torch.from_numpy(lat_sizes).float().log()
        mig_k = delta_mut_info / H_v
        mig = mig_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["mig_k"] = mig_k
            storer["mig"] = mig

        return mig

    def _axis_aligned_metric(self, sorted_mut_info, storer=None):
        """Compute the proposed axis aligned metrics."""
        numerator = (sorted_mut_info[:, 0] - sorted_mut_info[:, 1:].sum(dim=1)).clamp(min=0)
        aam_k = numerator / sorted_mut_info[:, 0]
        aam_k[torch.isnan(aam_k)] = 0
        aam = aam_k.mean()  # mean over factor of variations

        if storer is not None:
            storer["aam_k"] = aam_k
            storer["aam"] = aam

        return aam

    def _compute_q_zCx(self, dataloader):
        """Compute the empiricall disitribution of q(z|x).

        Parameter
        ---------
        dataloader: torch.utils.data.DataLoader
            Batch data iterator.

        Return
        ------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).
        """
        len_dataset = len(dataloader.dataset)
        latent_dim = self.model.latent_dim
        n_suff_stat = 2
        
        if self.model_type == 'ControlVAE':
           num_prop=self.model.num_prop 
           q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device) 
           q_wCx = torch.zeros(len_dataset, num_prop, n_suff_stat, device=self.device)
           n = 0
           with torch.no_grad():
                for x, label in dataloader:
                    batch_size = x.size(0)
                    idcs = slice(n, n + batch_size)
                    q_zCx[idcs, :, 0], q_wCx[idcs, :, 0], q_zCx[idcs, :, 1], q_wCx[idcs, :, 1], _ = self.model.encoder(x.to(self.device),label.to(self.device))                                      
                    n += batch_size

           params_zCX = q_zCx.unbind(-1)
           samples_zCx = self.model.reparameterize(*params_zCX)
           params_wCX = q_wCx.unbind(-1)
           samples_wCx = self.model.reparameterize(*params_wCX)
           return (samples_zCx,samples_wCx), (params_zCX,params_wCX)
        else:
           q_zCx = torch.zeros(len_dataset, latent_dim, n_suff_stat, device=self.device) 
           n = 0
           with torch.no_grad():
                for x, label in dataloader:
                    batch_size = x.size(0)
                    idcs = slice(n, n + batch_size)
                    q_zCx[idcs, :, 0], q_zCx[idcs, :, 1] = self.model.encoder(x.to(self.device))
                    n += batch_size

           params_zCX = q_zCx.unbind(-1)
           samples_zCx = self.model.reparameterize(*params_zCX)
           return samples_zCx, params_zCX            

    def compute_control_mse(self, dataloader):

           mse=[]
           R2=[]
           pred=[]
           labels=[]
           with torch.no_grad():
                for x, label in dataloader:
                   (reconstruct,y_reconstruct),_,_,_,_,_ = self.model(x.to(self.device),label.to(self.device))                                      
                   pred.append(y_reconstruct)
                   labels.append(label)
           pred=torch.cat(pred,dim=0).cpu().numpy()
           labels=torch.cat(labels,dim=0).cpu().numpy()
           if self.model.num_prop==3:
                labels = np.concatenate([labels[:,-4:-3],labels[:,-2:]],axis=-1)
           elif self.model.num_prop==2:
                labels = labels[:,-self.num_prop:]
           elif self.model.num_prop==4:
                labels = np.concatenate([labels[:,-3:-2],labels[:,-4],labels[:,-2:]],axis=-1)          
           for idx in range(y_reconstruct.size(1)):
               result=scoreReg(pred[:,idx],labels[:,idx])
               mse.append(result[0])
               R2.append(result[1])
           return (mse,R2)
       
    def compute_avgMI2(self, dataloader):
           all_score=np.zeros((self.model.latent_dim+self.model.num_prop, 6))
           latent=[]
           labels=[]
           with torch.no_grad():
                for x, label in dataloader:
                   z_mean,w_mean,z_std,w_std,_ = self.model.encoder(x.to(self.device),label.to(self.device)) 
                   mean=torch.cat([z_mean,w_mean],dim=-1)                                     
                   latent.append(mean)
                   labels.append(label)
           latent=torch.cat(latent,dim=0).cpu().numpy()
           labels=torch.cat(labels,dim=0).cpu().numpy()
           for z_idx in range(latent.shape[1]):
               for y_idx in range(label.shape[1]):
                   score=calc_MI(labels[:,y_idx],latent[:,z_idx])
                   print(score)
                   all_score[z_idx,y_idx]=score
           np.save('score.npy',all_score)        
           return all_score      

    def _estimate_latent_entropies(self, samples_zCx, params_zCX,
                                   n_samples=10000):
        r"""Estimate :math:`H(z_j) = E_{q(z_j)} [-log q(z_j)] = E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]`
        using the emperical distribution of :math:`p(x)`.

        Note
        ----
        - the expectation over the emperical distributio is: :math:`q(z) = 1/N sum_{n=1}^N q(z|x_n)`.
        - we assume that q(z|x) is factorial i.e. :math:`q(z|x) = \prod_j q(z_j|x)`.
        - computes numerically stable NLL: :math:`- log q(z) = log N - logsumexp_n=1^N log q(z|x_n)`.

        Parameters
        ----------
        samples_zCx: torch.tensor
            Tensor of shape (len_dataset, latent_dim) containing a sample of
            q(z|x) for every x in the dataset.

        params_zCX: tuple of torch.Tensor
            Sufficient statistics q(z|x) for each training example. E.g. for
            gaussian (mean, log_var) each of shape : (len_dataset, latent_dim).

        n_samples: int, optional
            Number of samples to use to estimate the entropies.

        Return
        ------
        H_z: torch.Tensor
            Tensor of shape (latent_dim) containing the marginal entropies H(z_j)
        """
        len_dataset, latent_dim = samples_zCx.shape
        device = samples_zCx.device
        H_z = torch.zeros(latent_dim, device=device)

        # sample from p(x)
        samples_x = torch.randperm(len_dataset, device=device)[:n_samples]
        # sample from p(z|x)
        samples_zCx = samples_zCx.index_select(0, samples_x).view(latent_dim, n_samples)

        mini_batch_size = 10
        samples_zCx = samples_zCx.expand(len_dataset, latent_dim, n_samples)
        mean = params_zCX[0].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_var = params_zCX[1].unsqueeze(-1).expand(len_dataset, latent_dim, n_samples)
        log_N = math.log(len_dataset)
        with trange(n_samples, leave=False, disable=self.is_progress_bar) as t:
            for k in range(0, n_samples, mini_batch_size):
                # log q(z_j|x) for n_samples
                idcs = slice(k, k + mini_batch_size)
                log_q_zCx = log_density_gaussian(samples_zCx[..., idcs],
                                                 mean[..., idcs],
                                                 log_var[..., idcs])
                # numerically stable log q(z_j) for n_samples:
                # log q(z_j) = -log N + logsumexp_{n=1}^N log q(z_j|x_n)
                # As we don't know q(z) we appoximate it with the monte carlo
                # expectation of q(z_j|x_n) over x. => fix a single z and look at
                # proba for every x to generate it. n_samples is not used here !
                log_q_z = -log_N + torch.logsumexp(log_q_zCx, dim=0, keepdim=False)
                # H(z_j) = E_{z_j}[- log q(z_j)]
                # mean over n_samples (i.e. dimesnion 1 because already summed over 0).
                H_z += (-log_q_z).sum(1)

                t.update(mini_batch_size)

        H_z /= n_samples

        return H_z

    def _estimate_H_zCv(self, samples_zCx, params_zCx, lat_sizes, lat_names):
        """Estimate conditional entropies :math:`H[z|v]`."""
        latent_dim = samples_zCx.size(-1)
        len_dataset = reduce((lambda x, y: x * y), lat_sizes) #1*3*6*40*32*32
        H_zCv = torch.zeros(len(lat_sizes), latent_dim, device=self.device) #[6,1]
        for i_fac_var, (lat_size, lat_name) in enumerate(zip(lat_sizes, lat_names)):
            idcs = [slice(None)] * len(lat_sizes)
            for i in range(lat_size):
                self.logger.info("Estimating conditional entropies for the {}th value of {}.".format(i, lat_name))
                idcs[i_fac_var] = i
                # samples from q(z,x|v)
                samples_zxCv = samples_zCx[idcs].contiguous().view(len_dataset // lat_size,
                                                                   latent_dim)
                params_zxCv = tuple(p[idcs].contiguous().view(len_dataset // lat_size, latent_dim)
                                    for p in params_zCx)

                H_zCv[i_fac_var] += self._estimate_latent_entropies(samples_zxCv, params_zxCv
                                                                    ) / lat_size
        return H_zCv

    
    def compute_aveMIG(self, dataloader):
        """Compute the average MIG for each pair of latent and property.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        """
        try:
            lat_sizes = dataloader.dataset.lat_sizes
            lat_names = dataloader.dataset.lat_names
        except AttributeError:
            raise ValueError("Dataset needs to have known true factors of variations to compute the metric. This does not seem to be the case for {}".format(type(dataloader.__dict__["dataset"]).__name__))


        score_results=torch.zeros(self.model.latent_dim+self.model.num_prop, self.model.num_prop)
        all_score=torch.zeros(self.model.latent_dim+self.model.num_prop, 6)

        self.logger.info("Computing the empirical distribution q(z|x).")
        samples_zCx, params_zCx = self._compute_q_zCx(dataloader)
        
        samples_zCx, samples_wCx = samples_zCx
        params_zCx, params_wCx = params_zCx            
        len_dataset, latent_dim = samples_zCx.shape
        len_dataset, num_prop = samples_wCx.shape
        # marginal entropy H(z_j)
        H_z = self._estimate_latent_entropies(samples_zCx, params_zCx)
        H_w = self._estimate_latent_entropies(samples_wCx, params_wCx)
        
        for latent_idx in range(self.model.latent_dim+self.model.num_prop):
                if latent_idx <self.model.latent_dim:
                    H_z_i=H_z.view(-1)[latent_idx]
                    samples_ziCx = samples_zCx[:,latent_idx].view(*lat_sizes, 1)
                    params_ziCx = tuple(p[:,latent_idx].view(*lat_sizes, 1) for p in params_zCx)
                     # conditional entropy H(zi|v)
                    H_zCv_i=self._estimate_H_zCv(samples_ziCx, params_ziCx, lat_sizes, lat_names) #[num_prop,1]
                    all_score[latent_idx]=-H_zCv_i.view(-1)+H_z_i.repeat(6)
                    score_results[latent_idx]=-H_zCv_i.view(-1)[-self.model.num_prop:]+H_z_i.repeat(self.model.num_prop)
                else:
                    w_latent_idx=latent_idx-self.model.latent_dim
                    H_w_i=H_w.view(-1)[w_latent_idx]
                    samples_wiCx = samples_wCx[:,w_latent_idx].view(*lat_sizes, 1) 
                    params_wiCx = tuple(p[:,w_latent_idx].view(*lat_sizes, 1) for p in params_wCx)                 
                    H_wCv_i=self._estimate_H_zCv(samples_wiCx, params_wiCx, lat_sizes, lat_names)
                    all_score[latent_idx]=-H_wCv_i.view(-1)+H_w_i.repeat(6)        
                    score_results[latent_idx]=-H_wCv_i.view(-1)[-self.model.num_prop:]+H_w_i.repeat(self.model.num_prop)        
                    
        self.logger.info("Estimating the marginal entropy.")
        
        #draw the mutual information image:
        np.save('score_results',all_score.cpu().numpy())
        
        
        #calculate the mse by comparing to the standard MIG:
        standard_mig=torch.zeros(self.model.latent_dim+self.model.num_prop, self.model.num_prop)
        for i in range(self.model.num_prop):
            standard_mig[i+self.model.latent_dim, i]=1
        avgMI_score_total=torch.norm(standard_mig-score_results)    
        avgMI_score_prop=torch.norm(standard_mig[self.model.latent_dim:]-score_results[self.model.latent_dim:]) 
       
  
        metrics = {'avgMI_total': avgMI_score_total.cpu().item(), 'avgMI_prop': avgMI_score_prop.cpu().item()}
        print(metrics)    
        #self.draw_mig(score_results)             
        return metrics, score_results
    
def draw_mig(mig):
       import matplotlib.pyplot as plt
       import seaborn as sns
       sns.set()
       mig=np.concatenate((mig[:-3].mean(0).reshape(1,6),mig[-3:]),axis=0)
       mig=np.concatenate((mig[:,2].reshape(4,1),mig[:,-2:]),axis=1)
       
       #import pandas as pd 
       #df = pd.DataFrame(mig)

       f, ax = plt.subplots(figsize=(4, 3))

       sns.heatmap(mig, annot=True, linewidths=1, ax=ax)

       label_y = ax.get_yticklabels()
       plt.setp(label_y, rotation=360, horizontalalignment='right')
       label_x = ax.get_xticklabels()
       plt.setp(label_x, rotation=45, horizontalalignment='right')
       #sns_plot.savefig(os.path.join(self.save_dir, 'avgMI'))