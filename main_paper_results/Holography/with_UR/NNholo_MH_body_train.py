import os
import time
t = time.process_time()
os.environ["DEV"] = "1"
os.environ["NEURODIFF_API_URL"] = "http://dev.neurodiff.io"
os.environ["NEURODIFF_API_KEY"] = 'tNaaIvvvdg72-c8VcTZRgpALsl0ns77ljEvxul6tG0E'
import warnings
warnings.filterwarnings("ignore")
#import dill
#print(dill.__version__)
from platform import python_version

import io
import pickle

import numpy
from NNholo_multihead_paper import *
print('NNHolo initialized')
from tqdm.auto import tqdm

#NEW directory
directory = 'results_2'
# Parent Directory path 
parent_dir = os.path.dirname(os.path.abspath(__file__))+"/MH_flattening_paper/flat_5_heads"
# Path 
path_results = os.path.join(parent_dir, directory) 

if os.path.exists(path_results) == False:
    os.mkdir(path_results) 
    print("New directory created")

#load_path = "/home/pedro/NNholo/dutch_heist/MH_fliying_dutchman/test_1_FALSE_VAC_less_epochs/NN_MH_heist_epochs_1500000"

phim_param=[3.0,2.0,1.5,1.08,1.0] #,0.8,0.75]#,0.72, 0.7,0.69, 0.68, 0.67, 0.66, 0.65, 0.645, 0.64, 0.635, 0.63, 0.625, 0.62, 0.615, 0.61, 0.605, 0.6]  #, 0.6, 0.58]

solver_nets = [32,32,32,32,128]
solver_head = [16,16]
V_nets = [32,32,32,128]
V_head = [64,64]
u_pts = 64

epochs = 1000000

sofT_list = []
for k in range(len(phim_param)):
        
    phim_str = str(phim_param[k])
    # Replace '.' with '_'
    phim_str = phim_str.replace('.', '_')

    sofT_path = os.path.dirname(os.path.abspath(__file__))+"/Data/Potentials_phim/SofTphiM" + phim_str + ".txt"

    sofT_list.append(sofT_path)
    

c = NNholo_multihead_heist(sofT_path_list=sofT_list, phim_list = phim_param,\
           saving_path = path_results, u_pts=u_pts, solver_nets=solver_nets, V_arch=V_nets, solver_head_arch=solver_head, \
           V_head_arch=V_head, actv_V_head=nn.SiLU, u_sampling='chebyshev2-noisy', n_batches_train=1, metric_flat_coef=500)#,load_path = load_path)
    
store_mse = Store_MSE_Loss()

scheduler = torch.optim.lr_scheduler.StepLR(c.adam, step_size=5000, gamma=0.985)
scheduler_cb = DoSchedulerStep(scheduler=scheduler)

potential_cb = BestValidationCallback()

c.solver.fit(max_epochs=epochs, 
             callbacks=[potential_cb, scheduler_cb], 
             tqdm_file= None) #tqdm(total=epochs, dynamic_ncols=True, desc='Epochs', unit='iteration', colour='#0afa9e'))
    

c.MH_save_results(f'{path_results}/NN_5MH_flat_metric')
#c.save_results(f'{path_results}/trained_NN_more_epochs')
c.plot_loss(save_fig=True)
plt.close()
c.plot_separate_losses(save_fig=True)
plt.close()
c.plot_potential(phim = phim_param, save_fig=True,show_legend = False)
plt.close()
c.plot_residuals_in_u(save_fig=True, max_bound=False)
plt.close()
#c.compare_to_yago(save_fig=True)
#c.save_results_new(path_results_now + "/trained_NN_phim_0_" + decimal_phim_now)

#do some stuff
c.plot_metric(save_fig = True)
plt.close

elapsed_time = time.process_time() - t
print(str(elapsed_time/3600) + ' horas' )
