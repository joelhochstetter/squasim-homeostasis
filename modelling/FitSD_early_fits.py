
'''
SD model: early time data

Run command:

parallel --max-procs 40 python3.12 SD_early_fits.py {1} 1 0 0 ::: {0..20000} &
parallel --max-procs 40 python3.12 SD_early_fits.py {1} 1 0 1 ::: {0..20000} &
parallel --max-procs 40 python3.12 SD_early_fits.py {1} 1 0 2 ::: {0..20000} &


Combine simulations:
python3.12 SD_early_fits.py -1 1 0 0 &
python3.12 SD_early_fits.py -1 1 0 1 &
python3.12 SD_early_fits.py -1 1 0 2 &

'''


#Imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.append("../../analysis/")
sys.path.append("../../sim/")
sys.path.append("../../fit/")
sys.path.append("../../plot/")


from tissue import *
import tissplot as tplt
import matplotlib.path as mplPath
import analysis
import datasets as ds 
import tiss_optimize as topt
import sim_model
import custom_metrics as cmet




#If we pass in all parameters as a string
if len(sys.argv) == 2:
    sys.argv = [sys.argv[0]] + sys.argv[1].split()

#Varying
i = int(sys.argv[1])

if len(sys.argv) > 2:
    ncores = int(sys.argv[2])
else:
    ncores = 1
    
if len(sys.argv) > 3:
    mode = int(sys.argv[3])
else:
    mode = 1
    
    

if len(sys.argv) > 4:
    didx = int(sys.argv[4])
else:
    didx = 0



dataset = ['data_ctr_M.pkl',    'data_ctr_F.pkl', 
                'data_castre.pkl', 'data_Ovx.pkl', 
                'data_Lgr6_M.pkl', 'data_Lgr6_F.pkl',
                'data_Notch1_M.pkl', 'data_Notch1_F.pkl']

if didx < 2:
    exp_file = '../experiment/homeo/' + dataset[didx]
elif didx == 2:
    exp_file = '../experiment/castre/' + dataset[didx]
elif didx == 3:
    exp_file = '../experiment/ovx/' + dataset[didx]    
elif didx  < 6:
    exp_file = '../experiment/lgr6/' + dataset[didx]    
elif didx < 8:
    exp_file = '../experiment/notch/' + dataset[didx]    
else:
    assert(0)


#Metrics
dists0  = {'dist': ['S2like.basalCS', 'S2like.supraCS'], 'dist_metric': 'lLL', 'name': 'dists0', 
        'exclude_times': {'S2like.basalCS': np.arange(1,8), 'S2like.supraCS': np.arange(1,8)}}
joints0 = {'custom_metrics': {'S2like.jointCS': cmet.logLL_joint}, 'name': 'joint0',
           'exclude_times': {'S2like.jointCS': np.arange(1,8)}}
dists1  = {'dist': ['S2like.basalCS', 'S2like.supraCS'], 'dist_metric': 'lLL', 'name': 'dists1', 
        'exclude_times': {'S2like.basalCS': np.arange(2,8), 'S2like.supraCS': np.arange(2,8)}}
joints1 = {'custom_metrics': {'S2like.jointCS': cmet.logLL_joint}, 'name': 'joint1',
           'exclude_times': {'S2like.jointCS': np.arange(2,8)}}

mets = [dists0, joints0, dists1, joints1]


compute_kwargs = {}

#Set-up optimiser
tscale_quant = 'S2like.basalCS'
savefolder = 'short_SD0D/'
s0 = topt.optimizer(sim_model.SD_0D, {'CVd_S':  np.arange(0.1, 1.01, 0.1),
                                      'CVd_D':  np.arange(0.1, 1.01, 0.1),
                                      'corrSS': np.arange(0.1, 1.01, 0.1),
                                      'r':      np.arange(0.1,1.01, 0.1),
                                      'rho': [0.4,0.55]},
                    params = {'tau': 100.0, 'corrSD': 0.0, 'corrSD': 0.0, 'corrDD': 0.0}, 
                    simtype = 'h', exp_file = exp_file,  exp_prefix = 'ctr', tshift = 2.0, metrics = mets,
                compute_kwargs = compute_kwargs, max_time = 30)
        

if i >= 0:
    s0.optimise(i = i,
        ncells = 1000, tsteps = 1000, seeds = np.arange(1), savefolder = savefolder,
        tscalevals = -np.arange(2,20,0.05), shedvals = -np.arange(0.2, 1.5, 0.1), 
        postopt_params = {'sbas_cv': np.arange(0.1, 1.01, 0.1)}, 
        use_saved_metrics = False, ncores = 1, save_every = 200, merge_during = i < 0,
        mergefirst = False, init_params = {'relaxsteps': 0, 'homeosteps': 1000}, sim_args = {},
        skip = 5, save_sweeps = False, par_seed = True,  nboots = 100, save_to_runspace = False,
        postopt_settings = {'tscale_by_quantity': tscale_quant, 'tscale_inc_times': np.arange(0,1)})
else:
    s0.merge_indexed_searchspace(np.arange(20000), ncells = 1000, tsteps = 1000, savefolder = savefolder)