'''

SD model - 0D

Run sweep with:
parallel --max-procs 50 python3.12 FitSD0D.py {1} 1 0 0 ::: {0..22000} & #M, SD
parallel --max-procs 50 python3.12 FitSD0D.py {1} 1 0 1 ::: {0..22000} & #F, SD

Combine sims with:
python3.12 FitSD0D.py -1000 1 0 0 &
python3.12 FitSD0D.py -1000 1 0 1 &


Run parallel simulated annealing with:
parallel --max-procs 40 python3.12 FitSD0D.py {1} 1 0 0 ::: {-40..-1} & #M, SD
parallel --max-procs 40 python3.12 FitSD0D.py {1} 1 0 1 ::: {-40..-1} & #F, SD

Combine sim anneal simulations with:
python3.12 FitSD0D.py -2000 1 0 0 &
python3.12 FitSD0D.py -2000 1 0 1 &

'''

#Imports
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../analysis/")
sys.path.append("../../sim/")
sys.path.append("../../fit/")
sys.path.append("../../plot/")


from tissue import *
import tissplot as tplt
import matplotlib.path as mplPath
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

# Using MF
if '_M' in exp_file:
    rho = 0.6
elif '_F' in exp_file:
    rho = 0.4

print(i, ncores, mode, didx)


#Metrics
ksdists  = {'dist': ['S2like.basalCS', 'S2like.supraCS'], 'dist_metric': 'ksd', 'name': 'ksd_b_sb'}
k1dists  = {'dist': ['S2like.basalCS'], 'dist_metric': 'ksd', 'name': 'ksd_bas'}
full = {'custom_metrics': {'S2like.fulljointCS': cmet.logLL_joint, 'Dlike.fulljointCS': cmet.logLL_joint}, 'name': 'full', 
        'exclude_times': {'S2like.fulljointCS': np.arange(6,7)}}

mets = [full, ksdists, k1dists]  #[full, ksdists, lsq_STD, lll_STD, full, means, dists, k1dists, joints, Dlo, xf]

compute_kwargs = {'Dlike': {'containsBasal': False}}
toCalculate = ['basalCS']


#Set-up optimiser
if mode == 0:
    savefolder = 'ctr_SD0D/'
    s0 = topt.sim_anneal(sim_model.SD_0D, {'r':   np.arange(0.1, 1.0, 0.05), 
                                          'CVd_S': np.arange(0.1, 1.01, 0.15),
                                          'CVd_D': np.arange(0.1, 1.01, 0.15),
                                          'corrSD': [0.0, 1.0], 
                                          'corrSS': [0.0, 0.75],
                                          'corrDD': [0.0, 0.75]},
                        params = {'tau': 500.0, 'rho': rho}, simtype = 'h',
                    exp_file = exp_file,  exp_prefix = 'ctr', tshift = 2.0, metrics = mets,
                    compute_kwargs = compute_kwargs, toCalculate = toCalculate,
                move_seed = np.abs(i),  use_linear = ['r', 'rho', 'CVd_S', 'CVd_D', 'corrSD', 'corrDD', 'corrSS'],
                move_sizes =     {'r': 0.05, 'rho':  0.025, 'CVd_S':  0.1, 'CVd_D':  0.1, 'corrSD':  0.1, 'corrDD':  0.1, 'corrSS':  0.1},
                free_param_min = {'r': 0.00, 'rho':   0.1,  'CVd_S':  0.0, 'CVd_D':  0.0, 'corrSD':  0.0, 'corrDD':  0.0, 'corrSS':  0.0},
                free_param_max = {'r': 1.00, 'rho':   1.0,  'CVd_S': 1.05, 'CVd_D': 1.05, 'corrSD': 1.05, 'corrDD': 1.05, 'corrSS': 1.05})


if i >= 0:
    s0.optimise(ncells = 4000, tsteps = 25000, seeds = np.arange(1), savefolder = savefolder,
        tscalevals = -np.arange(2,14,0.05), shedvals = -np.arange(0.2, 2.0, 0.1), 
        postopt_params = {'sbas_cv': np.arange(0.1, 1.01, 0.1)}, 
        use_saved_metrics = False, ncores = 0, save_every = 5, merge_during = i < 0,
        mergefirst = False, init_params = {'relaxsteps': 0, 'homeosteps': 1000}, sim_args = {},
        skip = 5, save_sweeps = False, par_seed = True, i = i*(i > 0) + -1*(i < 0), nboots = 0, save_to_runspace = False,
        postopt_settings = {'tscale_by_quantity': 'S2like.basalCS', 'tscale_inc_times': np.arange(2,6)})
elif i == -1000:
    s0.merge_indexed_searchspace(np.arange(22000), ncells = 4000, tsteps = 25000, savefolder = savefolder)

if i <= -1 and i > -1000:
    s0.runSA(ncells = 4000, tsteps = 25000, seeds = np.arange(1), savefolder = savefolder,
                tscalevals = -np.arange(2,14,0.05), shedvals = -np.arange(0.2, 2.0, 0.1), 
                postopt_params = {'sbas_cv': np.arange(0.1, 1.01, 0.1)}, merge_during = True,
                use_saved_metrics = True, ncores = ncores, save_every = 1, mergefirst = True,
                init_params = {'relaxsteps': 0, 'homeosteps': 1000}, sim_args = {'p0': 3.5, 'dt': 1e-2},
                SAsteps = 100, T0 = 10, start_idx = -1, skip = 5, save_sweeps = False,
                postopt_settings = {'tscale_by_quantity': 'S2like.basalCS', 'tscale_inc_times': np.arange(2,6)})

if i < -1000:
    s0.merge_seeded_searchspace(move_seeds = np.arange(200), ncells = 4000, tsteps = 25000, savefolder = savefolder)    
