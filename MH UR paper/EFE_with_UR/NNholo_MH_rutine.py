import os
import copy
from re import L
import dill 
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from copy import deepcopy
from tqdm.auto import tqdm
import torch.nn.functional as F
from ordered_set import OrderedSet
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import *
from neurodiffeq.solvers import BundleSolver1D
#from neurodiffeq.solvers import _diff_eqs_wrapper
from neurodiffeq.generators import BaseGenerator
from neurodiffeq.callbacks import ActionCallback 
#from neurodiffeq.generators import Generator1D, PredefinedGenerator
from generators import Generator1D, PredefinedGenerator
from neurodiffeq.conditions import BundleIVP, NoCondition, BundleDirichletBVP

large = 20
med = 16
small = 12

import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from neurodiffeq import diff

graphsize = (4, 4)
colors = ['#66bb6a', '#558ed5', '#dd6a63', '#dcd0ff', '#ffa726', '#8c5eff', '#f44336', '#00bcd4', '#ffc107', '#9c27b0']
params = {'axes.titlesize': small,
          'legend.fontsize': small,
          'figure.figsize': graphsize,
          'axes.labelsize': small,
          'axes.linewidth': 2,
          'xtick.labelsize': small,
          'xtick.color' : '#1D1717',
          'ytick.color' : '#1D1717',
          'ytick.labelsize': small,
          'axes.edgecolor':'#1D1717',
          'figure.titlesize': med,
          'axes.prop_cycle': cycler(color = colors),
          'text.usetex': False,
          'font.family': 'serif',  # Choose your font family (e.g., "serif", "sans-serif", "monospace")
          'font.serif': ['Times']}# Choose your font (e.g., "Times", "Arial", "Computer Modern Roman")}

# Define your custom colormap with #66bb6a
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', ['#66bb6a', '#1D1717'])
plt.rcParams.update(params)

IN_COLAB = torch.cuda.is_available()

def V_or(phi, phim = None):
    if phim == None:
        phim = 1.0
    else:
        phim = phim
    phiq = 10
    term1 = 6 * (2*phi)**2 * (8 + 2*(2*phi)**2/(phim)**2 - 3*(2*phi)**4 / phiq )**2
    term2 = (96 +8*(2*phi)**2 + (2*phi)**4/phim**2 - (2*phi)**6/phiq)**2
    return (1/4)*(1/768)*(term1 - term2)

def DV_or(phi, phim = None):
    if phim==None:
        phim = 1.0
    else:
        phim=phim
    phiq = 10
    return (-(1/(3*phim**4*phiq**2))*phi*(phiq**2*phi**4*(-9 + 2*phi**2)+
            2*phim**2*phiq*phi**4*(3*phiq+72*phi**2 - 10*phi**4) +
            phim**4*(12*phi**8*(-45 + 4*phi**2) + phiq**2*(9 + 4*phi**2) -
           4*phiq*phi**4*(-9 + 8*phi**2))))

def DDV_or(phi, phim=None):
    if phim==None:
        phim = 1.0
    else:
        phim=phim
    phiq = 10
    return (-3 - 4*phi**2 + phi**4 * ((15/phim**4) - (10/phim**2) - (60/phiq)) - ((176*phi**10)/(phiq**2)) + ((14*phi**6)/(3*phim**4 * phiq)) *(-72*phim**2 + 16*phim**4 - phiq) + ((60*phi**8)/(phim**2 * phiq**2))*(27*phim**2 + phiq))

from inspect import signature
import types


def _requires_closure(optimizer):
    # starting from torch v1.13, simple optimizers no longer have a `closure` argument
    closure_param = inspect.signature(optimizer.step).parameters.get('closure')
    return closure_param and closure_param.default == inspect._empty

class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the firs layer
        #self.mu =  torch.linspace(0,1, hidden_units[0])
        self.mu = nn.Parameter(torch.linspace(0,2, hidden_units[0]))
        #self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)
        self.sigma = torch.ones(hidden_units[0])*0.1

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        inputx = x[:,0].reshape(-1,1)
        #print(inputx.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x.shape)
            # Apply the custom operation after the first layer
            if i == 0:
                x = x * torch.exp(- (x - self.mu) ** 2 / self.sigma ** 2)

        # Output layer transformation
        x = self.fc_out(x)
        return x
    
class MeshGenerator(BaseGenerator):

    def __init__(self, g1, pg):

        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg

    def get_examples(self):

        u = self.g1.get_examples()
        u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)
        assert len(bundle_params[0].shape) == 1, "shape error, ask shuheng"
        n_params = len(bundle_params)

        bundle_params = torch.stack(bundle_params, dim=1)
        bundle_params = bundle_params.reshape(1, -1, n_params)

        uu, bb = torch.broadcast_tensors(u, bundle_params)
        uu = uu[:, :, 0].reshape(-1)
        bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]

        return uu, *bb

class minmaxScaler():
  def __init__(self, x):
    self.minx = x.min().detach().item()
    self.maxx = x.max().detach().item()

  def transform(self, x):
    return (x - self.minx)/(self.maxx - self.minx)
  
class DoSchedulerStep(ActionCallback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler

    def __call__(self, solver):
        self.scheduler.step()

class BestValidationCallback(ActionCallback):
    def __init__(self):
        super().__init__()
        self.best_potential = None

    def __call__(self, solver):
        if solver.lowest_loss is None or solver.metrics_history['r2_loss'][-1] <= solver.lowest_loss:
            self.best_potential = copy.deepcopy(solver.V_net)

class Store_MSE_Loss(ActionCallback):
    def __init__(self):
        super().__init__()
        self.mse_loss_history = []

    def __call__(self, solver):
        if solver.global_epoch % 10 == 0:
          for i in range(5):
            batch = self.generator['train'].get_examples()
            r = solver.get_residuals(*batch, to_numpy = True)
            self.mse_loss_history.append((np.array(r)**2).mean())

class CustomBundleSolver1D(BundleSolver1D):
    def __init__(self, sofT_list, phim_list, all_nets, V_net, g1, metric_flat_coef=100, TL=False, *args, **kwargs):
        
        self.n_heads = len(sofT_list)
        self.phim_param = phim_list
        self.func= ['$V_\Sigma$','$V_A$','$V_\phi$','$\Sigma$','$A$','$\phi$']
        #INPUT CURVES
        self.S_true_list = []
        self.T_true_list = []
        self.Sigma_uh_list = []
        self.Va_uh_list = []
        self.metric_flat_coef = metric_flat_coef


        for data_path in sofT_list:
            
            suffix_data_path = Path(data_path).suffix
            
            if suffix_data_path == ".csv":
                df_data = pd.read_csv(data_path, header=None).values
               # print('File is .csv')

                init_pt_curve = 55   #int(input('Init UV point S(T) ? (usually 55)'))
                S_true_1= torch.tensor(df_data[init_pt_curve:100:1,1])
                T_true_1= torch.tensor(df_data[init_pt_curve:100:1,0])
        
                S_true_3= torch.tensor(df_data[101:200:6,1])
                T_true_3= torch.tensor(df_data[101:200:6,0])
        
                S_true_4= torch.tensor(df_data[201::30,1])
                T_true_4= torch.tensor(df_data[201::30,0])
    
                S_true_lowest = torch.tensor([0])
                T_true_lowest = torch.tensor([0])
        
                self.S_true=torch.cat([S_true_1, S_true_3, S_true_4, S_true_lowest],dim=0)
                self.T_true=torch.cat([T_true_1, T_true_3, T_true_4, T_true_lowest],dim=0)
                
            if suffix_data_path == ".txt":
                df_data = pd.read_csv(data_path, sep=" ", header=None).values
               # print('File is .txt')
                
                init_pt_curve = 8    #int(input('Init UV point S(T) ? (usually 0)'))
                S_true_1= torch.tensor(df_data[init_pt_curve:90:2,1])
                T_true_1= torch.tensor(df_data[init_pt_curve:90:2,0])
                #phi_h_true_1=torch.tensor(phih_true[50:100:10])
        
                #S_true_2= torch.tensor(df_data[71:100:5,1])
                #T_true_2= torch.tensor(df_data[71:100:5,0])
        
                S_true_3= torch.tensor(df_data[91:165:3,1])
                T_true_3= torch.tensor(df_data[91:165:3,0])
                #phi_h_true_2=torch.tensor(phih_true[101:200:5])
        
                S_true_4= torch.tensor(df_data[166:-1:5,1])
                T_true_4= torch.tensor(df_data[166:-1:5,0])
    
                S_true_lowest = torch.tensor([0])
                T_true_lowest = torch.tensor([0])
        
                self.S_true=torch.cat([S_true_1, S_true_3, S_true_4, S_true_lowest],dim=0)
                self.T_true=torch.cat([T_true_1, T_true_3, T_true_4, T_true_lowest],dim=0)

            self.S_true_list.append(self.S_true)
            self.T_true_list.append(self.T_true)
            
            self.Sigma_uh_all = (self.S_true/np.pi)**(1/3)
            self.Va_uh_all = (-self.T_true*4*np.pi)

            self.Sigma_uh_list.append(self.Sigma_uh_all)
            self.Va_uh_list.append(self.Va_uh_all)
            self.gen_list = [] 
            
        print('length of s(T) curves:',[len(self.Sigma_uh_list[i]) for i in range(self.n_heads)])

        ################################

        self.phim_list = phim_list
        self.g1 = g1
        self.TL = TL
        self.equations = kwargs.pop('ode_system')
        #self.equations = ode_set

       # self.ode_set=[]
        #for head in range(self.n_heads):
         #   self.ode_set.append(self.equations[head])
        
        #self.V = kwargs.pop('V', None)
        self.alpha_2 = 0  #int(input('Loss coeff: '))


        self.ode_list = []
        for head in range(self.n_heads):
            print('head',head)
            #ode_system = kwargs.pop('ode_system')
            ode_system = self.equations[head]
            print('Inside solver __init()__:', ode_system)
            super().__init__(ode_system=ode_system, *args, **kwargs)
            self.ode_list.append(self.diff_eqs)
            #print('self.diff_eqs after super().__init__()', self.diff_eqs)
            self.pg = PredefinedGenerator(self.Sigma_uh_list[head], self.Va_uh_list[head])
            self.gen_list.append(MeshGenerator(self.g1, self.pg))

        #print('ode_list', self.ode_list, self.ode_list[0])
        self.all_nets = all_nets
        self.V_net = V_net
        self.head_add_loss = []
        self.metrics_history['r2_loss'] = []
        #self.metrics_history['head_add_loss'] = []
        self.metrics_history['add_loss'] = []
        self.metrics_history['DE_loss'] = []
        self.metrics_history['phi_max'] = []
       # self.metrics_fn = {"add_loss": 0.0,
        #                   "DE_loss": 0.0
        #}

        self.u = torch.linspace(0,1,50)
        sigma_list = []
        va_list = []     
        self.ode_list = []
        for head in range(self.n_heads):
            print('head',head)
            #ode_system = kwargs.pop('ode_system')
            ode_system = self.equations[head]
            print('Inside solver __init()__:', ode_system)
            super().__init__(ode_system=ode_system, *args, **kwargs)
            self.ode_list.append(self.diff_eqs)
            #print('self.diff_eqs after super().__init__()', self.diff_eqs)
            self.pg = PredefinedGenerator(self.Sigma_uh_list[head], self.Va_uh_list[head])
            self.gen_list.append(MeshGenerator(self.g1, self.pg))
            self.U,SIGMA = torch.meshgrid(self.u,self.Sigma_uh_list[head],indexing= 'ij')
            self.U,VA = torch.meshgrid(self.u,self.Va_uh_list[head],indexing= 'ij')
            sigma_list.append(SIGMA)
            va_list.append(VA)
        self.SIGMA = torch.stack(sigma_list)
        self.VA = torch.stack(va_list)

        #print('ode_list', self.ode_list, self.ode_list[0])
        self.all_nets = all_nets
        self.metrics_history['r2_loss'] = []
        self.metrics_history['add_loss'] = []
        self.metrics_history['phi_max'] = []

    def _set_loss_fn(self, criterion):
        pass

    def loss_fn(self,r,f,x):
        
        loss_r2 = (r**2).mean() 
        #self.metrics_history['phi_max'].append(f[5][-49: ].mean().detach().item())
        return loss_r2

    
    def additional_loss(self,r,f,x, head):
        
        if self.TL == False:
            k = 2.5
            thres = 400000
            add_coef = 0  #(1/2) + (1/2)*np.tanh(k*(self.global_epoch - 800000)/thres)
            
            p=f[5].reshape(-1,1)
            #print(cosa)
            V = self.V_net[head](p)
    #        DV = diff(self.V_net[head](p), p)
     #       DDV = diff(diff(self.V_net[head](p), p), p)
    
            phim=self.phim_list[head]
            #print('len(phi)',len(p))
            #print('phim', phim)
            #print('head', head)
            V_th = V_or(p,phim)
            DV_th = DV_or(p,phim)
            DDV_th = DDV_or(p, phim)
    
            add_V = ((V - V_th)**2).mean()
      #      add_DV = ((DV - DV_th)**2).mean()
       #     add_DDV = ((DDV - DDV_th)**2).mean()
    
            add_loss = add_V #+ add_DV + add_DDV
            #add_loss = (3 + DDV[-1])**2 + (0 - DV[-1])**2 + (3 + V[-1])**2
            #add_loss = 0.0  #Remove this to do heist #
            #print('add_loss', add_loss)
        elif self.TL == True:
            add_coef = 0
            add_loss= torch.tensor([0.0])

        add_loss_h = add_coef * add_loss

        if head == self.n_heads-1:
            add_loss_epoch = sum(self.head_add_loss)
            self.head_add_loss = []
            if torch.cuda.is_available():
                if add_loss_epoch==0:
                    add_loss_epoch=torch.tensor([0.0])
                self.metrics_history['add_loss'].append(add_loss_epoch.cpu().detach().numpy())
            else:
                if add_loss_epoch==0:
                    add_loss_epoch=torch.tensor([0.0])
                self.metrics_history['add_loss'].append(add_loss_epoch.detach().numpy())
        
        k = 0
        g_det = 0
        flat_metric = 1.0 *len(self.func)    #0
        if head == self.n_heads-1 and self.global_epoch% self.metric_flat_coef == 0 and self.TL == False:
            for f in range(len(self.func)):
                index = torch.linspace(0,len(self.Sigma_uh_list[k])-1,len(self.Sigma_uh_list[k]))
                phiM,U,IND = torch.meshgrid(torch.tensor(self.phim_param),self.u,index,indexing= 'ij') #[head, U, index]
                x = torch.cat([U.reshape(-1,1,1).squeeze(dim = 2),self.SIGMA.reshape(-1,1,1).squeeze(dim = 2),self.VA.reshape(-1,1,1).squeeze(dim = 2)]
                              ,dim = 1)  # input #
                H = self.all_nets[f,0].H_model(x)
                Omega = torch.cat([x,H],dim = 1)
                partial_omega = []
                for j in range(Omega.shape[1]):
                    partial_omega.append(diff(Omega[:,j],x,shape_check = False))
                partial_omega = torch.stack(partial_omega)
                #print(partial_omega.shape)  # [Coordinate index, sampling, metric dimension]
                partial_omega = torch.transpose(partial_omega,dim0 = 0,dim1 = 1)
                phiM,U,IND = torch.meshgrid(torch.tensor(self.phim_param),self.u,index,indexing= 'ij') #[head, U, index]
                g_MAT = torch.matmul(torch.transpose(partial_omega,dim0 = 1,dim1 = 2), partial_omega)
                g_det += torch.sqrt(torch.linalg.det(g_MAT)).reshape_as(U)
                flat_metric += torch.ones_like(g_det)
        #        cosa=f[5][0].reshape(-1,1)*0
        #       #print(cosa)
        #      V = self.V(cosa)
        #     DV = diff(self.V(cosa), cosa, shape_check=False)
        #    DDV = diff(diff(self.V(cosa),cosa,shape_check=False),cosa,shape_check=False)
         #   add_loss = (3 + DDV[-1])**2 + (0 - DV[-1])**2 + (3 + V[-1])**2
            self.dV = torch.sum(g_det-flat_metric)
            #print(self.dV)
            plain_metric =  self.dV*5e-8
        else:
            plain_metric =  torch.tensor([0.0])

        self.head_add_loss.append(add_loss_h + plain_metric)
                
        return plain_metric # + add_loss_h

    def _update_best(self, key):
        """Update ``self.lowest_loss`` and ``self.best_nets``
        if current training/validation loss is lower than ``self.lowest_loss``
        """
        current_loss = self.metrics_history['r2_loss'][-1]
        if (self.lowest_loss is None) or current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.best_nets = deepcopy(self.nets)


    def custom_epoch(self, key):
        r"""Run an epoch on train/valid points, update history, and perform an optimization step if key=='train'.

        :param key: {'train', 'valid'}; phase of the epoch
        :type key: str

        .. note::
            The optimization step is only performed after all batches are run.
        """
        if self.n_batches[key] <= 0:
            # XXX maybe we should append NaN to metric history?
            return
        self._phase = key
        
        tot_epoch_loss = 0.0
       # tot_epoch_add_loss = 0.0
       # tot_epoch_DE_loss = 0.0

        #batch_loss = 0.0

        
        loss = torch.tensor([0.0]) #, requires_grad=True) #added by me for multihead
        #add_loss = torch.tensor([0.0])
        #DE_loss = torch.tensor([0.0])

        
        metric_values = {name: 0.0 for name in self.metrics_fn}

        # Zero the gradient only once, before running the batches. Gradients of different batches are accumulated.
        if key == 'train' and not _requires_closure(self.optimizer):
            self.optimizer.zero_grad()

        # perform forward pass for all batches: a single graph is created and release in every iteration
        # see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/17

        for head in range(self.n_heads):
            #print('head', head)
            head_epoch_loss = 0.0
           # head_epoch_add_loss = 0.0
           # head_epoch_DE_loss = 0.0
                
            self.nets = self.all_nets[:,head]
            self.diff_eqs = self.ode_list[head]
            
            self.generator[key] = self.gen_list[head]
           # y = self.generator[key].get_examples()[1].detach().numpy()
           # plt.hist(y)
            #print(len(self.generator['train'].get_examples()))
        
            for batch_id in range(self.n_batches[key]):
                batch = self._generate_batch(key)
                
                #print(len(batch[0]))
               # print(self.n_batches[key])

                batch_loss = 0.0
              #  batch_add_loss = 0.0
               # batch_DE_loss = 0.0
    
                def closure(zero_grad=True):
                    nonlocal batch_loss
                    
                    if key == 'train' and zero_grad:
                        self.optimizer.zero_grad()
                    funcs = [
                        self.compute_func_val(n, c, *batch) for n, c in zip(self.nets, self.conditions)
                    ]
    
                    for name in self.metrics_fn:
                        value = self.metrics_fn[name](*funcs, *batch).item()
                        metric_values[name] += value
    
                    #CALLING THE EQUATIONS
    
                    residuals = self.diff_eqs(*funcs, *batch)
                    residuals = torch.cat(residuals, dim=1)
                   # print(residuals)
    
                    try:
                        DE_loss = self.loss_fn(residuals, funcs, batch)
                        add_loss = self.additional_loss(residuals, funcs, batch, head)
                        loss = DE_loss + add_loss
                        #self.metrics_history['add_loss'].append()
                        
                        #print('Head' + str(head) + ' loss ' + str(loss))
                        #if self.global_epoch %100:
                            #print('loss ',head,'=',loss)
                        
                    except TypeError as e:
                        warnings.warn(
                            "You might need to update your code. "
                            "Since v0.4.0; both `criterion` and `additional_loss` requires three inputs: "
                            "`residual`, `funcs`, and `coords`. See documentation for more.", FutureWarning)
                        raise e

                    # accumulate gradients before the current graph is collected as garbage
                    
                    #DOING BACKPROPAGATION
                    if key == 'train':
                        loss.backward()
                        batch_loss = loss.item()
                       # batch_add_loss = add_loss.item()
                       # batch_DE_loss = DE_loss.item()
                        #print('0',batch_loss)

                   # print(loss,add_loss,DE_loss)
                        
                    return loss  #, add_loss, DE_loss

                if key == 'train':

                        # Optimizer step will be performed only once outside the for-loop (i.e. after all batches).
                    closure(zero_grad=False)    #closure(zero_grad=False) was inside else in initial code

                    #print('key 2.batch')
    
                    head_epoch_loss += batch_loss
              #      head_epoch_add_loss += batch_add_loss
               #     head_epoch_DE_loss += batch_DE_loss
                    #print('head_loss', head_epoch_loss)
                    #head_epoch_loss += closure().item()

            if key == 'train':
                if _requires_closure(self.optimizer):
                    self._do_optimizer_step(closure=closure)
                    #print('optimizer step, key 1 head')
    
                ##else:
                    #closure(zero_grad=False)
                    #print('key 2.head')
    
                tot_epoch_loss += head_epoch_loss
            #    tot_epoch_add_loss += head_epoch_add_loss
             #   tot_epoch_DE_loss += head_epoch_DE_loss
             #   print('epoch losses: ', tot_epoch_loss,tot_epoch_add_loss,tot_epoch_DE_loss)

            else:
                tot_epoch_loss += closure()[0].item()
           #     tot_epoch_add_loss += closure()[1].item()
            #    tot_epoch_DE_loss += closure()[2].item()
                #print('key 3.head')

            # If validation is performed, update the best network with the validation loss
            # Otherwise, try to update the best network with the training loss
        #print(tot_epoch_loss)
        self.metrics_history['r2_loss'].append(tot_epoch_loss)

        
        if key == 'valid' or self.n_batches['valid'] == 0:
            self._update_best(key)

        # perform the optimizer step after all heads are run (if optimizer.step doesn't require `closure`)
        if key == 'train' and not _requires_closure(self.optimizer):
            self._do_optimizer_step()
            #print('optimizer step , key 4 head')
    
            #tot_epoch_loss += (head_epoch_loss / self.n_batches[key])

        # calculate the sum of all losses (one per head) and register to history
        
        self._update_history(tot_epoch_loss, 'loss', key)

     #   self.metrics_history['add_loss'].append(tot_epoch_add_loss)
      #  self.metrics_history['DE_loss'].append(tot_epoch_DE_loss)
        
        # calculate total metrics across heads (and averaged across batches) and register to history
        for name in self.metrics_fn:
            print(name)
            self._update_history(
                metric_values[name], name, key)


    def run_custom_epoch(self):
        r"""Run a training epoch, update history, and perform gradient descent."""
        self.custom_epoch('train')

    def fit(self, max_epochs, callbacks=(), tqdm_file='default', **kwargs):
        r"""Run multiple epochs of training and validation, update best loss at the end of each epoch.

        If ``callbacks`` is passed, callbacks are run, one at a time,
        after training, validating and updating best model.

        :param max_epochs: Number of epochs to run.
        :type max_epochs: int
        :param callbacks:
            A list of callback functions.
            Each function should accept the ``solver`` instance itself as its **only** argument.
        :rtype callbacks: list[callable]
        :param tqdm_file:
            File to write tqdm progress bar. If set to None, tqdm is not used at all.
            Defaults to ``sys.stderr``.
        :type tqdm_file: io.StringIO or _io.TextIOWrapper

        .. note::
            1. This method does not return solution, which is done in the ``.get_solution()`` method.
            2. A callback ``cb(solver)`` can set ``solver._stop_training`` to True to perform early stopping.
        """
        self._stop_training = False
        self._max_local_epoch = max_epochs

        self.callbacks = callbacks

        monitor = kwargs.pop('monitor', None)
        if monitor:
            warnings.warn("Passing `monitor` is deprecated, "
                          "use a MonitorCallback and pass a list of callbacks instead")
            callbacks = [monitor.to_callback()] + list(callbacks)
        if kwargs:
            raise ValueError(f'Unknown keyword argument(s): {list(kwargs.keys())}')  # pragma: no cover

        flag=False
        if str(tqdm_file) == 'default':
            bar = tqdm(
                total = max_epochs,
                desc='Training Progress',
                colour='blue',
                dynamic_ncols=True,
            )
        elif tqdm_file is not None:
            bar = tqdm_file
        else:
            flag=True
        
            

        for local_epoch in range(max_epochs):
             #stop training if self._stop_training is set to True by a callback
            if self._stop_training:
                break

            # register local epoch (starting from 1 instead of 0) so it can be accessed by callbacks
            self.local_epoch = local_epoch + 1
            #self.run_train_epoch()
            self.run_custom_epoch()
            self.run_valid_epoch()
            for cb in callbacks:
                cb(self)
            if not flag:
                bar.update(1)

# # IMPORT DATA
# df_data_yago_a1 = pd.read_csv("Data/1st order phim=1/A_yago_1.txt", sep=" ", header=None).values
# df_data_yago_sigma1 = pd.read_csv("Data/1st order phim=1/Sigma_yago_1.txt", sep=" ", header=None).values
# df_data_yago_phi1 = pd.read_csv("Data/1st order phim=1/phi_yago_1.txt", sep=" ", header=None).values

# A_yago1 = df_data_yago_a1[:, 1]
# u_yago1 = df_data_yago_a1[:, 0]
# Sigma_yago1 = df_data_yago_sigma1[:, 1]
# phi_yago1 = df_data_yago_phi1[:, 1]


# #point 2 (mid point)
# df_data_yago_a2 = pd.read_csv("Data/1st order phim=1/A_yago_2.txt", sep=" ", header=None).values
# df_data_yago_sigma2 = pd.read_csv("Data/1st order phim=1/Sigma_yago_2.txt", sep=" ", header=None).values
# df_data_yago_phi2 = pd.read_csv("Data/1st order phim=1/phi_yago_2.txt", sep=" ", header=None).values

# A_yago2 = df_data_yago_a2[:, 1]
# u_yago2 = df_data_yago_a2[:, 0]
# Sigma_yago2 = df_data_yago_sigma2[:, 1]
# phi_yago2 = df_data_yago_phi2[:, 1]

# #point 3 (left point)
# df_data_yago_a3 = pd.read_csv("Data/1st order phim=1/A_yago_3.txt", sep=" ", header=None).values
# df_data_yago_sigma3 = pd.read_csv("Data/1st order phim=1/Sigma_yago_3.txt", sep=" ", header=None).values
# df_data_yago_phi3 = pd.read_csv("Data/1st order phim=1/phi_yago_3.txt", sep=" ", header=None).values

# A_yago3 = df_data_yago_a3[:, 1]
# u_yago3 = df_data_yago_a3[:, 0]
# Sigma_yago3 = df_data_yago_sigma3[:, 1]
# phi_yago3 = df_data_yago_phi3[:, 1]


# Sigma_yago_all = [Sigma_yago1, Sigma_yago2, Sigma_yago3]
# A_yago_all = [A_yago1, A_yago2, A_yago3]
# phi_yago_all = [phi_yago1, phi_yago2, phi_yago3]

# u_yago=u_yago1 #same as u_yago2,3

# DA=[]
# [DA.append(np.gradient(A_yago_all[i],u_yago)) for i in range(3)]

# T_h=[]
# S_h=[]
# [T_h.append(DA[i][-1]/(-4*np.pi* u_yago[-1]**2)) for i in range(3)]

# [S_h.append((np.pi*Sigma_yago_all[i]**3)[-1]) for i in range(3)]

# #crossover
# #S_yago = [8.80154741176382, 1.2506823519737085, 0.17257280631479724] 
# #T_yago = [0.48423108257748665, 0.2883770837025976, 0.15629231178160138] 
# #phi_uh_yago= [0.6, 1, 1.13]

# #3of5
# S_yago = [8.939014410418975, 1.42689,  0.31734643273483476] 
# T_yago = [0.4867126785278015, 0.395869, 0.26805164866639136] 
# phi_uh_yago= [0.6, 1.4114516577290581, 1.528]

# Sigma_uh_yago=[]
# Va_uh_yago=[]

# for i in range(len(S_yago)):
#     Sigma_uh_yago.append((S_yago[i]/np.pi)**(1/3))
#     Va_uh_yago.append((-T_yago[i]*4*np.pi))


class NET(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET, self).__init__()
        self.H_model = H_model
        self.head_model = head_model
    
    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        return x
    
# MULTIHEAD NN FREEZE #
class NET_FREEZE(nn.Module):
    def __init__(self, H_model, head_model):
        super(NET_FREEZE, self).__init__()
        
        for param in H_model.parameters():
            param.requires_grad = False
        self.H_model = H_model
        self.head_model = head_model
        
        # Freeze the parameters of H_model
        #for param in self.H_model.parameters():
            #param.requires_grad = False
    
    def forward(self, x):
        x = self.H_model(x)
        x = self.head_model(x)
        return x

# DEFINE THE WHOLE RUTINE

    

class NNholo_multihead_heist():

    def __init__(self, saving_path, sofT_path_list=None, phim_list=None, delta = 0.0, curriculum = 1.0, u_pts = 48, \
        solver_nets=[32,32,32],V_arch = [16,16,16,16], head_type = 'FCNN', solver_head_arch = [32,16], \
        V_head_arch = [16,8], actv_V_head = nn.SiLU, u_sampling = 'chebyshev2-noisy', n_batches_train=1, \
        load_path = None, metric_flat_coef = 100, transfer_sofT_path = None, transfer_lr=1e-3, \
        load_head_TL = False, prev_optim_TL = False, unfreeze_H = False):

        self.delta = delta
        self.curriculum = curriculum
        self.path = saving_path
        self.phim_list = phim_list

        self.metric_flat_coef = metric_flat_coef

        pg_1 = torch.ones(100)
        pg_2 = torch.ones(100)
        #self.pg = PredefinedGenerator(self.Sigma_uh_all, self.Va_uh_all)
        self.pg = PredefinedGenerator(pg_1, pg_2)
        self.g1 = Generator1D(u_pts, 0, self.curriculum, method=u_sampling)
        self.g2 = Generator1D(16, 0, 1, method='equally-spaced')
        self.train_generator =  MeshGenerator(self.g1, self.pg)
        self.valid_generator =  MeshGenerator(self.g2, self.pg)
        self.n_batches_train = n_batches_train

        self.conditions = [
            NoCondition(),  # no condition on Vs
            BundleIVP(1, None, bundle_param_lookup=dict(u_0=1)), #condition on Va = -4 pi T
            BundleIVP(0, 1),   # Vphi(0) ==1
            BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)),  # Sigma_{u=0} = 1, Sigma_{u=1}=(S/pi)**(1/3)
            BundleDirichletBVP(0, 1, 1, 0),   # A (0) == 1  A(1)=0
            BundleIVP(0, 0),  #phi(0)=0 #BundleDirichletBVP(0, 0,1, phi_yago[-1])#
        ]
        
        if transfer_sofT_path != None:
            if load_path == None:
                raise ValueError("Must provide a pretrained model to use for TL")

            self.n_heads = 1
            self.MH_transfer(pretrained_path = load_path, sofT_path = transfer_sofT_path, lr=transfer_lr, load_head = load_head_TL, \
                prev_optim = prev_optim_TL, unfreeze_H=unfreeze_H)

            self.nets=self.nets_NEW
            self.V_net = self.V_NEW

        else:
            if sofT_path_list == None:
                raise ValueError("Must provide at least one S(T) curve for training")
            if phim_list == None:
                raise ValueError("Must provide list of phi_M associated to s(T) list")


            self.sofT_list = sofT_path_list
        

            self.solver_nets = solver_nets
            self.V_arch = V_arch
            self.solver_head_arch = solver_head_arch
            self.V_head_arch = V_head_arch

            self.n_heads = int(len(self.sofT_list))
                
            #path_SOFT = SofT_path_list
           # if self.n_heads != len(phiM_list):
            #    raise ValueError('The initial conditions and the lentgh of phiM must coincide with the number of heads')
            H = [FCNN(n_input_units=3, hidden_units=solver_nets[:-2], n_output_units = solver_nets[-1]) for _ in range(6)]
    
            if head_type == 'FCNN':
                heads = [[FCNN(n_input_units=solver_nets[-1], hidden_units=self.solver_head_arch, n_output_units = 1) for _ in range(self.n_heads)]for _ in range(len(H))]
                V_heads = [FCNN(n_input_units=self.V_arch[-1], hidden_units=self.V_head_arch, n_output_units = 1, actv=actv_V_head) for _ in range(self.n_heads)]
                
            elif head_type == 'Linear':
                heads = [[torch.nn.Linear(in_features=solver_nets[-1], out_features = 1) for _ in range(self.n_heads)]for _ in
                         range(len(H))]
                V_heads = [torch.nn.Linear(in_features=V_arch[-1], out_features = 1) for _ in range(self.n_heads)]
            
            #heads = [[torch.nn.Linear(in_features=solver_nets[-1], out_features = 1) for _ in range(self.n_heads)]for _ in
            #         range(len(H))]
    
            self.nets = np.ones([len(H),self.n_heads],dtype=nn.Module) # i -> equation, j -> head #
            for i in range(len(H)):
                for j in range(self.n_heads):
                    self.nets[i,j] = NET(H[i],heads[i][j])
    
            V_body = CustomNN(n_input_units = 1, hidden_units = self.V_arch[:-2] ,actv = nn.SiLU, n_output_units = self.V_arch[-1])
    
            #V_heads = [torch.nn.Linear(in_features=16, out_features = 1) for _ in range(self.n_heads)]
    
            self.V_net = []
    
            for i in range(len(V_heads)):
                self.V_net.append(NET(V_body,V_heads[i]))
            
            self.adam = torch.optim.Adam(OrderedSet([ p for q in range(self.n_heads) for net in list(self.nets[:,q])  + [self.V_net[q]] for p in net.parameters()]), lr=1e-3)#,  betas=(0.9, 0.99))
            
            if load_path != None:
                print('Loading the model...')
                path = load_path
                master_dict = torch.load(path, map_location=torch.device('cpu'))
                self.solver_nets = master_dict['solver_arch']
                self.solver_head_arch = master_dict['solver_head_arch']
                self.V_arch = master_dict['V_arch']
                self.V_head_arch = master_dict['V_head_arch']
                print('Solver architecture ' + str(self.solver_nets))
                print('Solver head architecture ' + str(self.solver_head_arch))
                print('V architecture ' + str(self.V_arch))
                print('V head architecture ' + str(self.V_head_arch))
                print('Creating the nets...')
                H = [FCNN(n_input_units=3, hidden_units=self.solver_nets[:-2], n_output_units = self.solver_nets[-1]) for _ in range(6)]
                heads = [[FCNN(n_input_units=self.solver_nets[-1], hidden_units=self.solver_head_arch,n_output_units = 1) for _ in range(self.n_heads)]for _ in range(len(H))]    
    
                self.nets = np.ones([len(H),self.n_heads],dtype=nn.Module) # i -> equation, j -> head #
                for i in range(len(H)):
                    #print(i)
                    for j in range(self.n_heads):
                        self.nets[i,j] = NET(H[i],heads[i][j])
                        
                print('Solver nets ready')

    
                V_body = CustomNN(n_input_units = 1, hidden_units = self.V_arch[:-2] ,actv = nn.SiLU, n_output_units = self.V_arch[-1])
    
                V_heads = [FCNN(n_input_units=self.V_arch[-1], hidden_units=self.V_head_arch,n_output_units = 1, actv=nn.SiLU) for _ in range(self.n_heads)]
    
                self.V_net = []
    
                for i in range(len(V_heads)):
                    self.V_net.append(NET(V_body,V_heads[i])) 
                    V_body.load_state_dict(master_dict['state_V_body'])
                  #  print('Load_state_dict done')
                for i in range(len(H)):
                    #print(i)
                    for j in range(self.n_heads):
                        H[i].load_state_dict(master_dict['body_dict_solver'][i])
                        heads[i][j].load_state_dict(master_dict['head_state'][i,j])
                        V_heads[j].load_state_dict(master_dict['V_head_state'][j])

                print('V-nets ready')
    
                self.adam = torch.optim.Adam(OrderedSet([ p for q in range(self.n_heads) for net in list(self.nets[:,q])  + [self.V_net[q]] for p in net.parameters()]), lr=1e-3)#,  betas=(0.9, 0.99))
                
                prev_loss = master_dict['loss']
                train_loss = master_dict['train_loss']
                add_loss = master_dict['add_loss']
                prev_optim = master_dict['optimizer']
                prev_epoch = master_dict['epoch']
                self.adam.load_state_dict(prev_optim)

    
            #self.V = self.V_net[0]
            self.eq_dict = {}
        
            self.generate_equations(n_heads = self.n_heads)
            #print(self.eq_dict)
            self.equation_list = []
            for head in range(self.n_heads):
                equations = self.eq_dict[f'equations_{head}']
                #print(equations)
                self.equation_list.append(equations)
        
            print('Equation list ready')
          #  print('Before we had:   ', [self.equations_0, self.equations_1, self.equations_2 ])
            
            self.solver = CustomBundleSolver1D( sofT_list = self.sofT_list,
                                                phim_list = self.phim_list,
                                                all_nets = self.nets,
                                                V_net = self.V_net,
                                               # ode_set = self.equations,
                                                g1 = self.g1,
                                                metric_flat_coef = self.metric_flat_coef,
                                                ode_system = self.equation_list, #[self.equations_0, self.equations_1, self.equations_2 ],  #self.ode_list,   #self.ode_list,  #self.equations
                                                conditions=self.conditions,
                                                t_min=self.delta,
                                                t_max=1,
                                                train_generator=self.train_generator,
                                                valid_generator=self.valid_generator,
                                                n_batches_train = n_batches_train,
                                                optimizer=self.adam,
                                                nets=self.nets,
                                                n_batches_valid=0,
                                                eq_param_index=()
                                            )
            if load_path != None:
                print('Loading data to the solver')
                self.solver.metrics_history['r2_loss'] = prev_loss
                self.solver.metrics_history['train_loss'] = train_loss
                self.solver.metrics_history['add_loss'] = add_loss
                
                del master_dict
                
                print('Ready')

            
    def sofT_curve(self):
        
        print('S_min: ', min(self.S_true))
        print('Length of input s(T) curve: ',  self.S_true.shape)

        #print('(S*,T*) = ', '(',S_h,',', T_h,')')
        #[print('(S*,T*)_%i'%(i+1) ,'= ', '(',S_yago[i],',', T_yago[i],')') for i in range(len(S_yago))]
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].scatter(self.T_true.cpu().detach().numpy(), self.S_true.cpu().detach().numpy())

        soverT3_x = self.T_true
        soverT3_y = self.S_true/(self.T_true**3)

        ax[1].scatter(soverT3_x.cpu().detach().numpy(), soverT3_y.cpu().detach().numpy())
        #plt.scatter((self.T_true),(self.S_true), color='k',s=15, label='true')
        #plt.xlabel('T')
        #plt.ylabel('s')
        plt.title(f'S_true shape: {len(self.S_true)}, Max T: {"{:.2f}".format(max(self.T_true))}')
        ax[0].set_xlabel('T')
        ax[0].set_ylabel('s')
        ax[1].set_xlabel('T')
        ax[1].set_ylabel('$s/T^3$')
        ax[0].legend()
        ax[1].legend()
        plt.show() 
        # [plt.scatter(T_yago[i], S_yago[i] ,s=15, color='r',label='(S*,T*) for  A(u), Sigma(u), phi(u) of Yago') for i in range(len(S_yago))]
        #plt.legend()
        plt.show() 

        plt.scatter((self.Va_uh_all.cpu().detach().numpy()), (self.Sigma_uh_all.cpu().detach().numpy()), color='k', s=15, label='true')
        #plt.hlines(0.78, -12.0, 0)
        plt.xlabel('$Va_h$')
        plt.ylabel('$\Sigma_h$')
        #[plt.scatter(Va_uh_yago[i], Sigma_uh_yago[i] ,s=15, color='r',label='(Sigma*,Va*) for  A(u), Sigma(u), phi(u) of Yago') for i in range(len(S_yago))]
        #plt.legend()
        plt.show()
        
        fig.savefig(f'{self.path}/sofT.png')
        
    def update_generator(self, curriculum = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, 0, curriculum, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
        
    def update_optimizer(self, lr = None):
        if lr == None:
            for g in self.adam.param_groups:
                print('Actual learning rate: ', g['lr'])
        else:
            for g in self.adam.param_groups:
                g['lr'] = lr
                print('Learning rate updated to: ', g['lr'])

    def custom_update_optimizer(self, order=None, epsilon=10000):

        for g in self.solver.optimizer.param_groups:
            if order !=None:
                g['lr'] = (10**order) * min(self.solver.metrics_history['train_loss'][-epsilon:])
                print('Learning rate updated to ', g['lr'], 'with order ', order)
            else:
                print('Learning rate: ', g['lr'])

    def custom_train(self, thres, callbacks=(), check_every=10000, step=None, epsilon=None, tqdm_file=None, print_plots=False):
        t = 0
        if step==None:
            step = check_every
    
        if epsilon == None:
            epsilon=int(0.05*step)
            
        while t==0:
            self.solver.fit(check_every, callbacks = callbacks, tqdm_file = tqdm_file)
            
            train = self.solver.metrics_history['train_loss']
            #a = np.log10(min(train[-init:-(init-step)]))
            #b =  np.log10(min(train[-step:]))
            #var = abs((10**a - 10**b)/(10**a)) * 100
            
            epsilon=int(0.05*step)
            init = len(train)-step
            end = init+step
            x_list = [init,end]
    
            a = np.log10(min(train[init:(init+epsilon)]))
            b =  np.log10(min(train[end-epsilon:end]))
            y_list= [a,b]
    
            slope = (10**b - 10**a)/(step)
            print('slope=',slope)

            if print_plots==True:
                fig,ax =plt.subplots(1,2,figsize=(6,3))
                ax[0].plot(np.linspace(init,end,len(train[init:end])), np.log10(train[init:end]))
                ax[0].plot(x_list,y_list, '-r', linewidth=1)
                ax[0].scatter(x_list, y_list, color='r', s=20, zorder=3)
                ax[0].axvline(init+epsilon, color='r', zorder=3)
                ax[0].axvline(end-epsilon, color='r', zorder=3)        
                ax[1].plot(np.log10(train))
                ax[1].axvline(init+epsilon, color='r', zorder=3)
                plt.show()
            
            if slope<0 and abs(slope)<thres:
                t+=1       
                print('Slope =', abs(slope), '< thres:',thres)

    def set_curriculum(self, start = 0.0, end = 1.0, valid_method = 'equally-spaced'):

        g1 = Generator1D(128, start, end, method='chebyshev2')
        g2 = Generator1D(16, 0, 1.0, method=valid_method)
        train_generator =  MeshGenerator(g1, self.pg)
        valid_generator =  MeshGenerator(g2, self.pg)

        self.solver.generator={'train': train_generator, 'valid': valid_generator}
    
    def equations(self, Vs, Va, Vp, Sigma, A, phi, u):
        
        #list = []
        #for head in range(self.n_heads):        
           # self.V = self.V_net[head]
        #self.V = self.V_net[self.head]
        VF = diff(self.V(phi), phi, shape_check= False)

        ORIGP_FLAG = 0

        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2

        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* self.V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)



        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)


        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*self.V(phi) + ORIGP_FLAG*V_or(phi)))
        
            #list.append([eq1, eq2, eq3, eq4 , eq5, eq6, eq7])
                    
        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]

    def equations_0(self, Vs, Va, Vp, Sigma, A, phi, u):

        V = self.V_net[0]
        VF = diff(V(phi), phi)#, shape_check= False)

        ORIGP_FLAG = 0

        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2

        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)



        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)


        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*V(phi) + ORIGP_FLAG*V_or(phi)))
        
            #list.append([eq1, eq2, eq3, eq4 , eq5, eq6, eq7])
                    
        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]

    def equations_1(self, Vs, Va, Vp, Sigma, A, phi, u):

        V = self.V_net[1]
        VF = diff(V(phi), phi)#, shape_check= False)
        
        ORIGP_FLAG = 0
        
        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2
        
        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)
        
        
        
        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)
        
        
        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*V(phi) + ORIGP_FLAG*V_or(phi)))
        
            #list.append([eq1, eq2, eq3, eq4 , eq5, eq6, eq7])
                    
        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]

    def equations_2(self, Vs, Va, Vp, Sigma, A, phi, u):

        V = self.V_net[2]
        VF = diff(V(phi), phi)#, shape_check= False)
        
        ORIGP_FLAG = 0
        
        # the equations
        eq1 = Vs - diff(Sigma, u, order=1)
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2
        
        eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* V(phi)  \
                                    + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                    + A * (8 * Sigma - 6 * u * Vs)
        
        
        
        eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
            + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)
        
        
        eq7 =  (u * Vs-Sigma) * \
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                2 * ((1-ORIGP_FLAG)*V(phi) + ORIGP_FLAG*V_or(phi)))
        
            #list.append([eq1, eq2, eq3, eq4 , eq5, eq6, eq7])
                    
        return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]

    def generate_equations(self, n_heads):    
        
        # Loop to create functions
        for head in range(n_heads):
            # Define a new function using a lambda or nested function
            def make_equations(head):
                
                def equations(self, Vs, Va, Vp, Sigma, A, phi, u):
                   # print('Now inside def equations(), head',head)

                    V = self.V_net[head]
                    VF = diff(V(phi), phi)#, shape_check= False)
                    
                    ORIGP_FLAG = 0
                    
                    # the equations
                    eq1 = Vs - diff(Sigma, u, order=1)
                    eq2 = Va - diff(A, u, order=1)
                    eq3 = Vp - diff(phi, u, order=1)
                    eq4 = diff(Vs, u,  order=1) + (2 / 3) *Sigma * Vp ** 2
                    
                    eq5 = (u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * ( (1-ORIGP_FLAG)* V(phi)  \
                                                + ORIGP_FLAG* V_or(phi) ) * Sigma  \
                                                + Va * (3 * u ** 2 * Vs - 5 * Sigma * u) \
                                                + A * (8 * Sigma - 6 * u * Vs)
                    
                    
                    
                    eq6 = u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (  (1-ORIGP_FLAG)*VF + ORIGP_FLAG* DV_or(phi)) \
                        + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs)
                    
                    
                    eq7 =  (u * Vs-Sigma) * \
                        ( u**2 * Sigma * Va + 2 * A * u**2 * Vs- 4 * u * A * Sigma) \
                        -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - \
                                            2 * ((1-ORIGP_FLAG)*V(phi) + ORIGP_FLAG*V_or(phi)))
                    
                        #list.append([eq1, eq2, eq3, eq4 , eq5, eq6, eq7])
                                
                    return [eq1, eq2, eq3, eq4 , eq5, eq6, eq7]
                return equations
        
            # Store the function in the dictionary
            self.eq_dict[f'equations_{head}'] = types.MethodType(make_equations(head), self)  #make_equations(head=head)
        print('Equations dictionary generated')
        return self.eq_dict

    def ode_set(self):
        ode_list = []
        for head in range(self.n_heads):
            self.V = self.V_net[head]
            eqs = self.equations
            ode_list.append(eqs)
        print(list(ode_list))

        return ode_list
    

    def get_loss(self):

        residuals = self.get_residuals()
        batch = [v.reshape(-1, 1) for v in self.valid_generator.get_examples()]
        funcs = [self.solver.compute_func_val(a, b, *batch) for a, b in zip(self.solver.nets, self.solver.conditions)]
        if IN_COLAB:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).detach().cpu().numpy()

        else:
            return self.solver.loss_fn(residuals, funcs, batch) + self.solver.additional_loss(residuals, funcs, batch).detach().numpy()
        
    def get_residuals(self, display = False):
        
        u, sigma, Va = self.valid_generator.get_examples()
        res = self.solver.get_residuals(u, sigma, Va, best=True)
        dim = int((res[0].shape[0])/16)
        res_eq = np.zeros((7, 16, dim)) 
        for i, r in enumerate(res):
            res_eq[i, :,:] =r.cpu().detach().reshape(16, dim)
        if display:
            print(f'Mean of residuals : {round((torch.cat(res) ** 2).mean().item(),9)}.')
        return res_eq
    
    def plot_residuals(self):
              
        residuals = self.get_residuals()
        
        fig, ax = plt.subplots(3,2, figsize=(6,18))
        ax = ax.flatten()

        vmax = 0.04

        levels = np.arange(0, vmax, .001)
        for eqn in np.arange(6):  
            im = ax[eqn].imshow( (np.abs(residuals[eqn,:,:].T)),  vmin=0, vmax=vmax, interpolation='bilinear', cmap=cmap)
            ax[eqn].contour(    (np.abs(residuals[eqn,:,:].T)), levels,   extend='both')
            ax[eqn].set_title(f"Eq{eqn+1}")
        # Add a colorbar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.9, 0.2, 0.05, 0.6])  # Adjust the position of the colorbar
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_loss(self, color=None, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', fontsize = 14, \
                  figsize=(8,6), thick=0.8, left_x_lim=0, save_fig = True):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        trace = self.solver.metrics_history
        fig1 = plt.figure(figsize=figsize)
        if color==None:
            plt.plot(np.log10(trace['train_loss']), label='train loss')
        else:
            plt.plot(np.log10(trace['train_loss']), label='train loss', color = color)
            
        if len(trace['valid_loss'])!=0:
            plt.plot(np.log10(trace['valid_loss']), label='validation loss')
        if 'train__res_eq1' in trace:
            for i in range(7): 
                plt.plot(np.log10(trace[f'train__res_eq{i+1}']), label = f'eq{i+1} residuals', alpha=0.6)
                
        
        # Customize the x-axis ticks and labels
        #custom_ticks = np.arange(0, 3e6, 500000)  # Define custom tick locations
        #custom_labels = ['0.5', '1', '1.5', '2', '2.5', '3']  # Define custom tick labels
        #plt.gca().set_xticks(custom_ticks)  # Set custom tick locations
        #plt.gca().set_xticklabels(custom_labels)  # Set custom tick labels
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        plt.xlabel(xlabel, fontsize = fontsize)
        #plt.ylabel('DE Residual Square Loss')
        plt.ylabel(ylabel, fontsize = fontsize)
        #plt.xlim(left=left_x_lim,right=3e6)
        #plt.grid()
        #plt.legend(loc='upper right')
        #plt.tight_layout()
        plt.show()
        
        print('Min loss: ', min(trace['train_loss']))
        
        #fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}_prettier.png')
        #fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}_prettier.eps')
        if save_fig==True:
            fig1.savefig(f'{self.path}/loss_epoch {trained_epochs}.pdf')

    def plot_separate_losses(self, xlabel= 'epochs', ylabel = r'$\log_{10}\mathcal{L}$', \
              figsize=(8,6), thick=0.8, left_x_lim=0, save_fig = False):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        loss_add = self.solver.metrics_history['add_loss']
        #print('loss_add', loss_add)
        loss_train = self.solver.metrics_history['train_loss']
        #loss_DE = self.solver.metrics_history['DE_loss']
      #  if len(loss_add)!=len(loss_train):
        ratio = self.solver.metric_flat_coef #int(len(loss_train)/len(loss_add))
        
        renorm_loss_add = [loss_add[0]]
        for i in range(len(loss_add)-1):
            if i % ratio == 0:
               # print(i,'diff from 0')
                renorm_loss_add.append(loss_add[i+1])
            else:
                #print(renorm_add_loss)
                renorm_loss_add.append(renorm_loss_add[i])
        
    #    renorm_loss_add = []
     #   for i in range(int(len(loss_add)/ratio)):
         #   print(loss_add[i+1])
      #      [renorm_loss_add.append(loss_add[i*ratio+1]) for _ in range(ratio)] 
            #loss_add[i] = [loss_add[i] for _ in range(ratio)]
        
       # print('renorm', renorm_loss_add)
        loss_DE = np.hstack(loss_train) - np.hstack(renorm_loss_add)
        renorm_loss_add = np.hstack(renorm_loss_add)
        #print('renorm', renorm_loss_add)

       # print(len(loss_DE))
       # else:
        #    loss_DE = np.hstack(loss_train) - np.hstack(loss_add)
         #   renorm_loss_add = np.hstack(loss_add)
        #print(loss_train)
        #print(loss_add)
        #a = torch.stack(loss_add)
       # b = torch.tensor(loss_train)
        #print(a)
        #c = b-a
        
        fig2 = plt.figure(figsize=figsize)
        #plt.plot(loss_train, label='train')
        plt.plot(loss_DE, label='DE')
        plt.plot(renorm_loss_add, label = 'Flat_add')
        plt.legend()
        plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        
        print('Min total loss: ', min(loss_train))
        #print('Min DE loss: ', min(loss_DE))
        #print('Min additional loss: ', min(loss_add))
                                          
        fig, ax =plt.subplots(1,3, figsize=(10,3))
        ax[0].plot(loss_train)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Train loss')
        ax[1].plot(loss_DE)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('DE loss')
        ax[2].plot(renorm_loss_add)
        ax[2].set_yscale('log')
        ax[2].set_ylabel('Add loss')
        plt.show()
        
        if save_fig==True:
            fig2.savefig(f'{self.path}/all_losses_epoch_{trained_epochs}.pdf')
            
    def plot_metric(self, save_fig = True):

        #start = time.time()
        # Try to compute the metric #
        heads = np.linspace(0, self.n_heads, self.n_heads)
        phim_param = self.phim_list
       
        func= ['$V_\Sigma$','$V_A$','$V_\phi$','$\Sigma$','$A$','$\phi$']
        func_txt = ['Vs','Va','Vp','Sigma', 'A', 'phi']


        # Prepare the batch #
        u = torch.linspace(0,1,50)
        sigma_list = []
        va_list = []
        for k in range(len(heads)):
            U,SIGMA = torch.meshgrid(u,self.solver.Sigma_uh_list[k],indexing= 'ij')
            U,VA = torch.meshgrid(u,self.solver.Va_uh_list[k],indexing= 'ij')
            sigma_list.append(SIGMA)
            va_list.append(VA)
        SIGMA = torch.stack(sigma_list)
        VA = torch.stack(va_list)
       
        for f in range(len(func)):
            index = torch.linspace(0,len(self.solver.Sigma_uh_list[k])-1,len(self.solver.Sigma_uh_list[k]))
            phiM,U,IND = torch.meshgrid(torch.tensor(phim_param),u,index,indexing= 'ij') #[head, U, index]
            x = torch.cat([U.reshape(-1,1,1).squeeze(dim = 2),SIGMA.reshape(-1,1,1).squeeze(dim = 2),VA.reshape(-1,1,1).squeeze(dim = 2)]
                          ,dim = 1)  # input #
            H = self.solver.all_nets[f,0].H_model(x)
            Omega = torch.cat([x,H],dim = 1)
            gmunu = torch.zeros([len(x),len(x)])
            partial_omega = []
            for j in range(Omega.shape[1]):
                partial_omega.append(diff(Omega[:,j],x,shape_check = False))
            partial_omega = torch.stack(partial_omega)
            #print(partial_omega.shape)  # [Coordinate index, sampling, metric dimension]
            partial_omega = torch.transpose(partial_omega,dim0 = 0,dim1 = 1)
            phiM,U,IND = torch.meshgrid(torch.tensor(phim_param),u,index,indexing= 'ij') #[head, U, index]
            g_MAT = torch.matmul(torch.transpose(partial_omega,dim0 = 1,dim1 = 2), partial_omega)
            print('g_MAT',g_MAT.shape,'det', torch.linalg.det(g_MAT).shape, 'U', U.shape)

            g_det = torch.sqrt(torch.linalg.det(g_MAT)).reshape_as(U)
            point = 10
            for k in range(len(heads)):
                fig1,ax = plt.subplots(1,2,figsize=(10,5))
                colormesh = ax[0].pcolormesh(U[k,:,:].cpu().detach().numpy(),IND[k,:,:].cpu().detach().numpy(),g_det[k,:,:].cpu().detach().numpy()
                                             ,cmap = 'rainbow')
                plt.colorbar(colormesh,ax = ax[0])
                fig1.suptitle('Function ' + str(func[f]) + ' for $\phi_M = $' + str(phim_param[k]))
                ax[0].hlines(point,0,1,linewidth = 2.5, color = 'black',linestyle = 'dashed')
                ax[0].set_title('$\sqrt{|g|}$')
                ax[0].set_xlabel('U')
                ax[0].set_ylabel('index')
                ax[1].plot(self.solver.Va_uh_list[k].cpu().detach().numpy(),self.solver.Sigma_uh_list[k].cpu().detach().numpy(),'.k')
                ax[1].plot(self.solver.Va_uh_list[k][point].cpu().detach().numpy(),self.solver.Sigma_uh_list[k][point].cpu().detach().numpy(),'or')
                ax[1].set_xlabel('$V_{a,h}$')
                ax[1].set_ylabel('$\Sigma_{h}$')
                plt.show()

                if save_fig==True:
                   
                    fig1.savefig(f'{self.path}/metric_{func_txt[f]}_phiM_{str(phim_param[k])}_epoch_{self.solver.global_epoch}.pdf')
   
            fig2 = plt.figure(figsize = (10,5))
            ax = fig2.add_subplot(111,projection='3d')
            scatter = ax.scatter(U.cpu().detach().numpy(),IND.cpu().detach().numpy(),phiM.cpu().detach(),
                                 c=g_det.cpu().detach().numpy(),cmap='rainbow',alpha = 0.4)

            ax.set_xlabel('u')
            ax.set_ylabel('index')
            ax.set_zlabel('$\phi_M$')
            plt.colorbar(scatter,ax = ax)
            ax.set_title('Function ' + str(func[f]))
            plt.show()
       
        #end = time.time()
        #print(end-start) 
    def plot_potential(self, phim, sampling_pts = 100, save_fig = True, best = False, show_legend=True, log_scale=False):
        
        trained_epochs = len(self.solver.metrics_history['train_loss'])
        
        u = np.linspace(0, 1, sampling_pts)
        
        phi_list = []
        V_list = []
        phi_h_th_list = []
        phi_h_max_list = []
        V_nn_min_list = []
        
        fig = plt.figure(figsize=(8,8))
        #col_th = ['dodgerblue', 'limegreen', 'indianred', 'gray', '#ff7f0e', 'orchid']
        #col_new = ['b','green', 'r', 'k', '#ff7f0e' ,'purple']
        col_th=[]
        for c in range(self.n_heads):
            color = 'C'+str(c)
            col_th.append(color)
        col_new=col_th

        for head in range(self.n_heads):
            
            self.solver.best_nets = self.nets[:,head]
            solution = self.solver.get_solution(best=True)

            phi_h = np.ones(self.solver.S_true_list[head].shape)
            true_phi_h = np.ones(self.solver.S_true_list[head].shape)
            u_max = np.ones(self.solver.S_true_list[head].shape)
    
            for i,S in enumerate(self.solver.S_true_list[head]):
                T=self.solver.T_true_list[head][i]
            #    print(i,S,T)
                Sigma_v = (S/np.pi)**(1/3)
                Va_v = (-T*4*np.pi)
                Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
                Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
                Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True)
                phi_h[i] = phi.max()
                true_phi_h[i] = phi[-1]
                i_max = phi.argmax()
                u_max[i] = u[i_max]
            print('max phi_h= ',max(phi_h))
            phi_h_max_list.append(max(phi_h))
    
            # Define the domain of input phi
            phi=torch.reshape(torch.linspace(0,max(phi_h),100),[100,1])
            phi_th = torch.linspace(0,3,100).cpu().detach().numpy()
            phi = torch.Tensor(phi)
            phi.requires_grad = True
            qphi = phi.cpu().detach().numpy().reshape(-1,)
            qphi.shape
            phi_list.append(qphi)
    
            #Vv = potential_cb.best_potential(phi) #potential_cb.best_potential(phi)
            #DVv = diff(potential_cb.best_potential(phi), phi, shape_check= False)
            #DDVv = diff(potential_cb.best_potential(phi), phi, order=2, shape_check= False)
            Vv = self.V_net[head](phi)
            V_list.append(Vv)
            potentialVphi=pd.DataFrame(phi.cpu().detach().numpy())
            potentialVVv=pd.DataFrame(Vv.cpu().detach().numpy())
            #potentialDVVv=pd.DataFrame(DVv.cpu().detach().numpy())
            #potentialDDVVv=pd.DataFrame(DDVv.cpu().detach().numpy())
    
            py_listphi=phi.tolist()
            py_listV=Vv.tolist()
            #py_listDV=DVv.tolist()
            #py_listDDV=DDVv.tolist()
            #potentialV=pd.DataFrame(list(zip(py_listphi,py_listV,py_listDV,py_listDDV)),columns=['Phi','V','DV','DDV'])
            potentialV=pd.DataFrame(list(zip(py_listphi,py_listV)),columns=['Phi','V'])
            potentialV['Phi']=potentialV['Phi'].str[0]
            potentialV['V']=potentialV['V'].str[0]
            #potentialV['DV']=potentialV['DV'].str[0]
            #potentialV['DDV']=potentialV['DDV'].str[0]
            potentialV.to_csv(f'{self.path}/V_head_{head}_epoch_{trained_epochs}.csv', index=False)

            plt.plot(phi_th, V_or(phi_th, phim[head]).reshape(-1,1), label=fr'Theory, $\phi_M=%.2f$'%phim[head], color = col_th[head], linestyle = 'dashed', linewidth = 2.5, zorder=1)
            plt.plot(qphi, Vv.cpu().detach(), label=f'NN head {head}', color = col_new[head])

            phi_h_th = phi_th[np.argmin(V_or(phi_th,phim[head]))]
            phi_h_th_list.append(phi_h_th)
            plt.axvline(phi_h_th, color = col_th[head], linestyle = 'dashed', label='$\phi_{h,th}=%.2f$'%phi_h_th, linewidth=1)
            plt.axvline(max(phi_h), color = col_new[head], linestyle = 'solid', linewidth = 1, label ='$\phi_{h,NN}=$%.2f'%max(phi_h))
            plt.axhline(min(V_or(phi_th, phim[head])), color = col_th[head], linestyle ='dotted', label='$V_{th}^{min}=%.2f$'%min(V_or(phi_th,phim[head])), linewidth=1)  
            V_nn_interp = interp1d(phi.cpu().detach().numpy()[:,0], Vv.cpu().detach().numpy()[:,0])
            V_nn_h = V_nn_interp(phi_h)
            ind = np.argmax(phi_h)
            V_nn_min_list.append(V_nn_h[ind])
            plt.axhline(V_nn_h[ind], color = col_new[head], linestyle ='solid', linewidth = 1, label='$V_{NN}^{min}=%.2f$'%V_nn_h[ind])
            #V_nn_min_list.append(V_nn_h[int(np.where(phi_h==max(phi_h))[0])])
            #plt.axhline(V_nn_h[int(np.where(phi_h==max(phi_h))[0])], color = col_new[head], linestyle ='solid', linewidth = 1, label='$V_{NN}^{min}=%.2f$'%V_nn_h[int(np.where(phi_h==max(phi_h))[0])])

        plt.xlim(0.0, max([phi_h_th+0.1, max(phi_h) + 0.1]))
        plt.ylim(min([min(V_or(phi_th, phim[-1]))-1, V_nn_min_list[-1]-1]),-2)
        if show_legend==True:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=self.n_heads)
        if log_scale==True:
            plt.yscale('symlog')

        if save_fig == True:
            fig.savefig(f'{self.path}/V_all_heads_epoch {trained_epochs}.pdf')
        plt.show()

        if self.n_heads>1:
            #if self.n_heads<=3:
             #   rows = 1
            #elif 3<self.n_heads<=6:
             #   rows=2
            #elif 6<self.n_heads<=9:
             #   rows = 3
            rows = 1   
            fig2,ax = plt.subplots(rows,self.n_heads,figsize = (20,4))
            
            for head in range(self.n_heads):
                
                ax[head].plot(qphi, V_or(qphi, phim[head]).reshape(-1,1), label=fr'Theory, $\phi_M=%.2f$'%phim[head], color = col_th[head], linestyle = 'dashed', linewidth = 3, zorder = 1)
                ax[head].plot(phi_list[head], V_list[head].cpu().detach(), label=f'NN head {head}', color = col_new[head], zorder=2)
                ax[head].axvline(phi_h_th_list[head], color = col_th[head], linestyle = 'dashed', label='$\phi_{h,th}=%.2f$'%phi_h_th_list[head], linewidth = 1)
                ax[head].axvline(phi_h_max_list[head], color = col_new[head], linestyle = 'solid', linewidth = 1, label ='$\phi_{h,NN}=$%.2f'%phi_h_max_list[head])
                ax[head].axhline(min(V_or(qphi, phim[head])), color = col_th[head], linestyle ='dotted', label='$V_{th}^{min}=%.2f$'%min(V_or(qphi,phim[head])), linewidth = 1)  
                ax[head].axhline(V_nn_min_list[head], color = col_new[head], linestyle ='solid', linewidth = 1, label='$V_{NN}^{min}=%.2f$'%V_nn_min_list[head])

                #ax[head].legend(loc='lower left')
                ax[head].set_xlim(0.0, phi_h_max_list[head]+0.1)
                ax[head].set_ylim(V_nn_min_list[head]-1,-2)
    
            if save_fig == True:
                fig2.savefig(f'{self.path}/V_heads_epoch_{trained_epochs}.pdf')
            plt.show()
     

      #  cosa=torch.reshape(torch.tensor([0.0]),[-1,1])
        #print(cosa)
       # cosa.requires_grad_(requires_grad = True)
        #print('V(0)',self.V(cosa))
        #print('DV(0)',diff(self.V(cosa), cosa, shape_check=False))
        #print('DDV(0)', diff(diff(self.V(cosa), cosa, shape_check=False), cosa, shape_check=False))
        
        
    def plot_colored_sofT(self, phiM_chosen, colormap = 'vidris', fontsize = 14, n_fontsize = 14, \
                          dot_size=10, thick=0.8, figsize = (8,6)):
        
        u = np.linspace(0.0001, 1, 100)
        phi_h = np.ones(self.S_true.shape)
        solution = self.solver.get_solution(best=True)
        
        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True)
            phi_h[i] = phi.max()
            #true_phi_h[i] = phi[-1]
            #i_max = phi.argmax()
            #u_max[i] = u[i_max]

        #plt.scatter(phi_h, self.T_true.cpu(), color=color)
        #plt.xlabel(r'$\phi_H$')
        #plt.ylabel('T')
        #plt.show()

        #plt.scatter(phi_h, self.S_true.cpu())
        #plt.xlabel('phih')
        #plt.ylabel('S')
        #plt.show()

        step = 1
        fig = plt.figure(figsize=figsize)
        plt.scatter((self.T_true.cpu()[::step]),(self.S_true.cpu()[::step]), c=phi_h[::step], s=dot_size, cmap=colormap)
        #plt.hlines(1.5028, min(T_true), max(T_true))
        colorbar = plt.colorbar()
        colorbar.set_label(r'$\phi_H$', fontsize = fontsize)
        colorbar.ax.yaxis.label.set_rotation(0) 
        plt.xlabel(r'$T/\Lambda$', fontsize=fontsize)
        plt.ylabel(r'$S/\Lambda^3$', fontsize=fontsize)
        plt.xticks(fontsize=n_fontsize)  # Adjust the font size as needed
        plt.yticks(fontsize=n_fontsize)
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        #plt.legend()
        fig.savefig(f'/Users/pablo/Desktop/NNholo/To run/January (2024)/Plots paper/colored s(T) by phi_h (phiM={phiM_chosen}).pdf')

        plt.show()
        
    def plot_s_over_phi_h(self, phiM_chosen, color = 'k', fontsize = 14, n_fontsize = 14, \
                          dot_size=10, thick=0.8, figsize = (8,6)):

        u = np.linspace(0.0001, 1, 100)
        phi_h = np.ones(self.S_true.shape)
        solution = self.solver.get_solution(best=True)

        for i,S in enumerate(self.S_true):
            T=self.T_true[i]
            Sigma_v = (S/np.pi)**(1/3)
            Va_v = (-T*4*np.pi)
            Sigma_uh = Sigma_v.cpu().detach().numpy()*np.ones_like(u)
            Va_uh = Va_v.cpu().detach().numpy()*np.ones_like(u)
            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_uh,  Va_uh, to_numpy=True)
            phi_h[i] = phi.max()
            #true_phi_h[i] = phi[-1]
            #i_max = phi.argmax()
            #u_max[i] = u[i_max]

        #plt.scatter(phi_h, self.T_true.cpu())
        #plt.xlabel('phih')
        #plt.ylabel('T')
        #plt.show()
        fig = plt.figure(figsize=figsize)
        plt.scatter(phi_h, self.S_true.cpu(), color = color, s=dot_size)
        plt.xlabel(r'$\phi_H$', fontsize= fontsize)
        plt.ylabel(r'$S/\Lambda^3$', fontsize= fontsize)
        #plt.legend(fontsize = 10)
        #plt.xlim(left=min(phi_h),right=max(phi_h))
        #plt.ylim(bottom=min(self.S_true.cpu()),top=max(self.S_true.cpu()))

        plt.xticks(fontsize=n_fontsize)  # Adjust the font size as needed
        plt.yticks(fontsize=n_fontsize)
        plt.gca().spines['top'].set_linewidth(thick)  # Adjust the thickness as needed
        plt.gca().spines['right'].set_linewidth(thick)
        plt.gca().spines['bottom'].set_linewidth(thick)
        plt.gca().spines['left'].set_linewidth(thick)
        plt.show()

        fig.savefig(f'/Users/pablo/Desktop/NNholo/To run/January (2024)/Plots paper/s(phi_h) (phiM={phiM_chosen}).pdf')

        
    def compare_to_yago(self, fontsize = 14, legend_fontsize=14, n_fontsize=14, wspace=0.5, yago_linewidth = 3, yago_style = '--', save_fig = True):
        
        for i in range(3):
            
            fig, ax = plt.subplots(1,2, figsize=(16,7))
            pt=i+1
            #u = np.linspace(0.0001, 1, len(u_yago))
            u = u_yago
            #S_yago_sol = 1.42689 
            #T_yago_sol = 0.395869
            #phi_uh_yago=1.4114516577290581

            #print('here', phi_uh_yago)

            S_tt=S_yago[pt-1]
            T_tt=T_yago[pt-1]

            Sigma_h = (S_tt*np.ones_like(u)/np.pi)**(1/3)
            Va_h = (-T_tt*np.ones_like(u)*4*np.pi)

            #Sigma_h = .78*np.ones_like(u)
            #Va_h = -10.0*np.ones_like(u)

            solution = self.solver.get_solution(best=True)

            Vs, Va, Vp, Sigma, A, phi = solution(u, Sigma_h,  Va_h, to_numpy=True)

            print('Point %i' %pt, '; phi_h_yago = %f' %phi_uh_yago[pt-1])

            ax[0].plot(u, Sigma, 'r-', label=r'$\tilde{\Sigma}_{NN}$', zorder=2)
            ax[0].plot(u_yago , Sigma_yago_all[pt-1], 'r', linestyle = yago_style, label=r'$\tilde{\Sigma}_{th}$', linewidth = yago_linewidth, zorder=1)

            ax[0].plot(u, A, 'b-', label=r'$\tilde{A}_{NN}$', zorder=2)
            ax[0].plot(u_yago, A_yago_all[pt-1], 'b', linestyle = yago_style, label = r'$\tilde{A}_{th}$', linewidth = yago_linewidth, zorder=1) 

            ax[0].plot(u, phi, 'g-', label=r'$\phi_{NN}$', zorder=2)
            ax[0].plot(u_yago, phi_yago_all[pt-1], 'g', linestyle = yago_style, label=r'$\phi_{th}$', linewidth = yago_linewidth, zorder=1)
            #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
           
            ax[0].set_xlabel('u', fontsize = fontsize)
            ax[0].set_ylabel('ODEs solutions', fontsize = fontsize)
            ax[0].set_xlim(-0.05,1.05)
            ax[0].set_ylim(-0.05,max(max(Sigma),max(A),max(phi))+0.05)
            
            #ax[0].set_title(title,fontsize = fontsize)
            ax[0].legend(fontsize = legend_fontsize)
            #plt.grid()
            #plt.savefig('solution3.png')

            #ax[1].title('Squared residuals')
            MSE_Sigma = (1/len(Sigma))*sum((Sigma_yago_all[pt-1]-Sigma)**2)
            MSE_A = (1/len(A))*sum((A_yago_all[pt-1]-A)**2)
            MSE_phi = (1/len(phi))*sum((phi_yago_all[pt-1]-phi)**2)
            
            rel_err_Sigma = (abs(Sigma_yago_all[pt-1]-Sigma)/Sigma_yago_all[pt-1])
            rel_err_A = (abs(A_yago_all[pt-1]-A)/A_yago_all[pt-1])
            rel_err_phi = (abs(phi_yago_all[pt-1]-phi)/phi_yago_all[pt-1])
            
            ax[1].plot(u,(Sigma_yago_all[pt-1]-Sigma)**2, color= 'r', label=r'$MSE_{\tilde{\Sigma}}=%.1e$'%MSE_Sigma)
            ax[1].plot(u,(A_yago_all[pt-1]-A)**2, color= 'b', label=r'$MSE_{\tilde{A}}=%.1e$'%MSE_A)
            ax[1].plot(u,(phi_yago_all[pt-1]-phi)**2, color= 'g', label=r'$MSE_{\phi}=%.1e$'%MSE_phi)
            #ax[1].ticklabel_format(axis='y', style='sci', scilimits=(1,1))
            ax[1].set_xlabel('u',fontsize = fontsize)
            ax[1].set_ylabel(r'(theory$-$NN)$^2$',fontsize = fontsize)
            ax[1].set_xlim(0,1)
            ax[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax[1].yaxis.get_offset_text().set_fontsize(n_fontsize)

            ax[1].legend(fontsize = legend_fontsize)
            
            title = r"$(T_{:.0f}, S_{:.0f})=({:.2f}, {:.2f})$".format(i+1,i+1,T_tt, S_tt)
            plt.suptitle(title, fontsize = fontsize)
            
            for ax in ax:
                ax.tick_params(axis='both', which='major', labelsize=n_fontsize)  # Adjust the font size
            plt.subplots_adjust(wspace=wspace)  # Increase or decrease the value to adjust the spacing
            plt.show()
            
            if save_fig == True:
                fig.savefig(f'{self.path}/DE solution_pt_{i+1}_(phiM=1,best).pdf')
            
    def plot_V_theory(self, phim = 1):
        
        col=['k','b','g','m','r','y']
        
        phim_param = phim
        #labels=['1st order s(T)','Crossover s(T)']
        plt.figure()
        phi_mins=[]
        for i in range(len(phim_param)):
            phim=phim_param[i]
            phi_th=np.linspace(0,3,250)
            V_th=V_or(phi_th, phim)
            V_phi_min=V_th[0]
            cnt=0
            for j in range(len(phi_th)):
                if V_th[j]<V_phi_min:
                    V_phi_min=V_th[j]
                    phi_min=phi_th[j]
                    cnt=j
            phi_mins.append(phi_min)
            plt.plot(phi_th[0:cnt+1],V_th[0:cnt+1],color=col[i],label='$\phi_M$=%.2f'%phim)
            plt.axvline(phi_min, linestyle = 'dotted', color=col[i])
            #plt.ylim(-8,-2)
            plt.xlim(0,max(phi_mins)+0.1)
            plt.grid()
            plt.legend()
        plt.show()
        
    def plot_residuals_in_u(self, max_bound = 0.05, print_overbound = False, save_fig=True):
       ### RESIDUAL PLOTS
        # PICK ANY
        u = torch.linspace(0, 1, 250)
        loss_bound = np.sqrt(min(self.solver.metrics_history['train_loss']))

        
        if max_bound != False:
            bottom, top = -max_bound, max_bound

        for head in range(self.n_heads):

            self.solver.best_nets = self.nets[:,head]
            self.solver.diff_eqs = self.solver.ode_list[head]
                        
            fig = plt.figure()
            for q,i in enumerate(zip(self.solver.Sigma_uh_list[head], self.solver.Va_uh_list[head])):
                Sigma_h = i[0].cpu().detach()*torch.ones_like(u)
                Va_h = i[1].cpu().detach()*torch.ones_like(u)
    
                if q % 5 == 0:
                    #print(q)
                    res1 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[0].cpu().detach().numpy()
                    res2 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[1].cpu().detach().numpy()
                    res3 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[2].cpu().detach().numpy()
                    res = [res1,res2,res3]
    
                    plt.plot(u.cpu().detach().numpy(), res1 , 'r-' ,  alpha=0.1,label='Vs Eq1')
                    plt.plot(u.cpu().detach().numpy(), res2 , 'b-',  alpha=0.1, label='Va Eq2')
                    plt.plot(u.cpu().detach().numpy(), res3 ,'g-', alpha=0.1,label='Vp Eq3')
                    
                    if q==0:
                        plt.axhline(loss_bound, linestyle = '--', color='k')
                        plt.axhline(-loss_bound, linestyle = '--', color='k')
                        plt.legend()
                    
                    if print_overbound == True:
                        for k in range(len(res)):
                            for j in range(len(res[k])):
                                if abs(res[k][j]) > top:
                                    print('For u =', u[j], ', residual(eq.',k,')=',abs(res[k][j]))
                                
            if max_bound != False:    
                plt.ylim(bottom,top)
            plt.xlim(0,1)
            plt.xlabel('u')
            plt.ylabel('DE residual')
            plt.title(f'head {head}')
                    
            plt.show()
        
            if save_fig==True:
                fig.savefig(f'{self.path}/DE_res_1-3_head_{head}.pdf')
                
        for head in range(self.n_heads):

            self.solver.best_nets = self.nets[:,head]
            self.solver.diff_eqs = self.solver.ode_list[head]
            
            fig2 = plt.figure()
            for q,i in enumerate(zip(self.solver.Sigma_uh_list[head], self.solver.Va_uh_list[head])):
    
                Sigma_h = i[0].cpu().detach()*torch.ones_like(u)
                Va_h = i[1].cpu().detach()*torch.ones_like(u)
                    
                if q % 5 ==0:
                    #print(q)
                    
                    res4 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[3].cpu().detach().numpy()
                    res5 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[4].cpu().detach().numpy()
                    res6 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[5].cpu().detach().numpy()
                    res7 = self.solver.get_residuals(u,Sigma_h , Va_h,  best=True)[6].cpu().detach().numpy()
                    res = [res4,res5,res6, res7]
                       
                    plt.plot(u.cpu().detach().numpy(), res4  , 'b-', alpha=0.1 ,label='Eq4')
                    plt.plot(u.cpu().detach().numpy(), res5 ,   'r-', alpha=0.1, label='Eq5' )
                    plt.plot(u.cpu().detach().numpy(), res6  ,  'g-', alpha=0.1, label='Eq6' )
                    plt.plot(u.cpu().detach().numpy(), res7  ,  'k-', alpha=0.1, label='Eq7' )
                    
                    if q==0:
                        plt.axhline(loss_bound, linestyle = '--', color='k')
                        plt.axhline(-loss_bound, linestyle = '--', color='k')
                        plt.legend()
                    
                    if print_overbound == True:
                        for k in range(len(res)):
                            for j in range(len(res[k])):
                                if abs(res[k][j]) > top:
                                    print('For u =', u[j], ', residual(eq.',k,')=',abs(res[k][j]))
                                
            if max_bound != False:    
                plt.ylim(bottom,top)
            plt.xlim(0,1)
            plt.xlabel('u')
            plt.ylabel('DE residual')
            plt.title(f'head {head}')

            plt.show()
                
            if save_fig==True:
                fig2.savefig(f'{self.path}/DE_res_4-7_head_{head}.pdf')
        
    def render(self):

        self.plot_loss()
        self.plot_residuals()
        self.plot_result()
        #self.compare_to_yago()

    def save_results(self, path):

        self.solver.save(path=path)
        with open(path, 'rb') as file:
            data = dill.load(file)
        os.remove(path)
        try:
            data['V_best'] = self.solver.callbacks[0].best_potential.state_dict()
            data['V_latest'] = self.V.state_dict()

        except:
            data['V_latest'] = self.V.state_dict()
        with open(path, 'wb') as file:
            dill.dump(data, file)

    def load_results(self, path):

        with open(path, 'rb') as file:
            data = dill.load(file)

        self.saved_data = data   
        
        try:     
            self.V.load_state_dict(data['V_best'])
        except:
            self.V.load_state_dict(data['V_latest'])

        train_generator = data['generator']['train']
        valid_generator = data['generator']['valid']
        de_system = data['diff_eqs']
        cond = data['conditions']
        nets = data['nets']
        #print(nets)
        best_nets = data['best_nets']
        #print(best_nets)
        train_loss = data['train_loss_history']
        valid_loss = data['valid_loss_history']
        optimizer = data['optimizer_class'](OrderedSet([p for net in data['nets'] + [self.V] for p in net.parameters()]))
        optimizer.load_state_dict(data['optimizer_state'])
        if data['generator']['train'].generator:
            #t_min = data['generator']['train'].generator.__dict__['g1'].__dict__['t_min']
            #t_max = data['generator']['train'].generator.__dict__['g1'].__dict__['t_max']
            t_min = 0.0
            t_max = 1.0
        else:
            #t_min = data['generator']['train'].__dict__['g1'].__dict__['t_min']
            #t_max = data['generator']['train'].__dict__['g1'].__dict__['t_max']
            t_min = 0.0
            t_max = 1.0

        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=cond,
                                            t_min=t_min,
                                            t_max=t_max,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=optimizer,
                                            nets=nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )

        if best_nets != None:
            self.solver.best_nets = best_nets
        self.solver.metrics_history['train_loss'] = train_loss
        self.solver.metrics_history['valid_loss'] = valid_loss
        self.solver.diff_eqs_source = data['diff_equation_details']['equation']
        
    def save_results_new(self, path):

        nets_state = []
        best_nets_state = []
        for i in range(len(self.solver.nets)):
            nets_state.append(self.solver.nets[i].state_dict())
            best_nets_state.append(self.solver.best_nets[i].state_dict())
        #print(nets_state)
        #print(c.solver.metrics_history['r2_loss'])
        #print(c.solver.global_epoch)
        #print(c.adam.state_dict())
        state = {'epoch': self.solver.global_epoch, 'state_dict_V': self.V.state_dict(),'state_dict_solver': nets_state,'state_best_nets': best_nets_state,
                     'optimizer': self.adam.state_dict(), 'loss': self.solver.metrics_history['r2_loss'],
                'train_loss': self.solver.metrics_history['train_loss']}
        torch.save(state,path)
        print('Model succesfully saved')

    def load_results_new(self, path):
        master_dict = torch.load(path, map_location=torch.device('cpu'))
        self.V.load_state_dict(master_dict['state_dict_V'])
        self.adam.load_state_dict(master_dict['optimizer'])
        self.solver = CustomBundleSolver1D( ode_system=self.equations,
                                            conditions=self.conditions,
                                            t_min=0.0,
                                            t_max=1.0,
                                            train_generator=self.train_generator,
                                            valid_generator=self.valid_generator,
                                            optimizer=self.adam,
                                            nets=self.nets,
                                            n_batches_valid=0,
                                            eq_param_index=(),
                                            V = self.V
                                        )
        self.solver.metrics_history['r2_loss'] = master_dict['loss']
        self.solver.metrics_history['train_loss'] = master_dict['train_loss']
        self.solver.metris_history['add_loss'] = master_dict['add_loss']
        self.solver.best_nets = np.ones_like(self.solver.nets)
        for i in range(len(self.solver.nets)):
            self.solver.nets[i].load_state_dict(master_dict['state_dict_solver'][i])
            if master_dict['state_best_nets'][i] != None:
                self.solver.best_nets[i] = self.solver.nets[i]
                self.solver.best_nets[i].load_state_dict(master_dict['state_best_nets'][i])
        print(self.update_optimizer())
        print('Model succesfully loaded')
        
    def MH_save_results(self,path):
        body_state = []
        head_state = np.ones([len(self.nets[:,0]),int(self.n_heads)],dtype = object)
        for n in range(len(self.nets[:,0])):
            body_state.append(self.nets[n,0].H_model.state_dict())
        for i in range(len(self.nets[:,0])):
            for j in range(len(self.nets[0,:])):
                head_state[i,j] = self.nets[i,j].head_model.state_dict()
        V_body_state = self.V_net[0].H_model.state_dict()
        V_head_state = []
        for i in range(len(self.V_net)):
            V_head_state.append(self.V_net[i].head_model.state_dict())
        state = {'epoch': len(self.solver.metrics_history['train_loss']), 'state_V_body': V_body_state,'V_head_state': V_head_state,'body_dict_solver':
                 body_state,'head_state': head_state,'optimizer': self.solver.optimizer.state_dict(), 'loss':self.solver.metrics_history['r2_loss'] ,
                'train_loss':self.solver.metrics_history['train_loss'],
                 'add_loss': self.solver.metrics_history['add_loss'],
                'solver_arch':self.solver_nets,
                 'solver_head_arch':self.solver_head_arch,
                 'V_arch':self.V_arch,
                 'V_head_arch':self.V_head_arch
                }
        torch.save(state,path+'_epochs_'+str(len(self.solver.metrics_history['train_loss'])))
        print('Model succesfully saved')
        
    def MH_transfer(self, pretrained_path, sofT_path, lr=5e-5, prev_optim = True, load_head=False, unfreeze_H=False):
        
        path_results = self.path
        path = pretrained_path
        print('Loading dictionary...')
        master_dict = torch.load(path,  map_location=torch.device('cpu'))
        print('Dictionary loaded correctly')

        self.solver_nets = master_dict['solver_arch']
        self.solver_head_arch = master_dict['solver_head_arch']
        self.V_arch = master_dict['V_arch']
        self.V_head_arch = master_dict['V_head_arch']
        
        V_body_NEW  = CustomNN(n_input_units = 1, hidden_units = self.V_arch[:-2] ,actv = nn.SiLU, n_output_units = self.V_arch[-1])
        V_body_NEW.load_state_dict(master_dict['state_V_body'])

        H_NEW = [FCNN(n_input_units=3, hidden_units=self.solver_nets[:-2],n_output_units = self.solver_nets[-1]) for _ in range(6)]
        for i in range(len(H_NEW)):
            H_NEW[i].load_state_dict(master_dict['body_dict_solver'][i])

        head_NEW = [FCNN(n_input_units=self.solver_nets[-1], hidden_units=self.solver_head_arch, n_output_units = 1)for _ in range(len(H_NEW))]
        self.nets_NEW = np.ones([len(H_NEW),1],dtype=nn.Module) # i -> equation, j -> head #

        V_head_NEW = FCNN(n_input_units=self.V_arch[-1], hidden_units=self.V_head_arch, n_output_units = 1)

        if load_head == True:
            V_head_NEW.load_state_dict(master_dict['V_head_state'][0])
            for i in range(len(H_NEW)):
                head_NEW[i].load_state_dict(master_dict['head_state'][i,0])

        if unfreeze_H==False:
            for i in range(len(H_NEW)):
                self.nets_NEW[i] = NET_FREEZE(H_NEW[i],head_NEW[i])

            self.V_NEW = [NET_FREEZE(V_body_NEW,V_head_NEW)]
        else:
            for i in range(len(H_NEW)):
                self.nets_NEW[i] = NET(H_NEW[i],head_NEW[i])

            self.V_NEW = [NET(V_body_NEW,V_head_NEW)]
            
        #plt.imshow(new_H[0].NN[2].weight.detach().numpy())
        #plt.show()
        #print(head_NEW[0].NN[2].weight.detach().numpy())
        #plt.plot(head_NEW[0].NN[2].weight.detach().numpy()[0,:],'.b')
        #plt.show()
        #plt.imshow(V_NEW.H_model.layers[2].weight.detach().numpy())
        #plt.show()
        
        prev_loss = master_dict['loss']
        train_loss = master_dict['train_loss']
        add_loss = master_dict['add_loss']
        prev_epoch = master_dict['epoch']
        prev_optim_state = master_dict['optimizer']
     #   prev_optim = 
      #  for g in prev_optim.param_groups:
       #     lr_prev = g['lr']

        self.adam = torch.optim.Adam(OrderedSet([ p for net in list(self.nets_NEW[:,0])  + [self.V_NEW[0]] for p in net.parameters()]), lr=lr)
        if prev_optim == True:
            self.adam.load_state_dict(prev_optim_state)

        self.V = self.V_NEW
        
        self.eq_dict = {}
        self.V_net = self.V_NEW
        self.generate_equations(n_heads = 1)
        self.equations_TL = []

        for head in range(1):
            equations = self.eq_dict[f'equations_{head}']
            #print(equations)
            self.equations_TL.append(equations)
            
            #print(prev_optim)
        #optim.load_state_dict(prev_optim)
        #optim = torch.optim.Adam(OrderedSet([p for net in list(nets_NEW[:]) + [V_NEW] for p in net.parameters()]), \
        #                        lr=1e-3)
        
        self.solver = CustomBundleSolver1D(sofT_list = [sofT_path],
                                              phim_list = self.phim_list,
                                              all_nets = self.nets_NEW,
                                              V_net = self.V_net,
                                              g1 = self.g1,
                                              TL = True,
                                              ode_system = self.equations_TL, #[self.equations_0, self.equations_1, self.equations_2 ],  #self.ode_list,   #self.ode_list,  #self.equations
                                              conditions=self.conditions,
                                              t_min=self.delta,
                                              t_max=1,
                                              train_generator=self.train_generator,
                                              valid_generator=self.valid_generator,
                                              n_batches_train = self.n_batches_train,
                                              optimizer=self.adam,
                                              nets=self.nets_NEW,
                                              n_batches_valid=0,
                                              eq_param_index=()
                                             )
        
        self.solver.metrics_history['r2_loss'] = prev_loss
        self.solver.metrics_history['train_loss'] = train_loss
        self.solver.metrics_history['add_loss'] = add_loss
        
        print('Ready to transfer learn')
