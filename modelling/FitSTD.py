'''
Fit STD model

Running simulated annealing with cST, cTD, cvS fixed
or use a grid search parameter sweep
An extensive parmaeter sweep using both methods was performed to obtain the best fit
The grid search was used for bootstrapping to estimate uncertainties on parameters

Usage:
python3.12 FitSTD.py i ncores mode sex ncells cTD cST cvS metmode cvD tau hilln


We repeated simulated annealing procedure to find the best fit values
with different fixed parameters



#Comment out the model use
mode=1 #Use 2D
#mode=0 #Use 0D


#Males 
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.0 -1.0 0.5 0 ::: {-120..-101} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.5 -1.0 0.5 0 ::: {-140..-121} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.0  1.0 0.5 0 ::: {-160..-141} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.5  1.0 0.5 0 ::: {-180..-161} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 1.0  1.0 0.5 0 ::: {-200..-181} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.0 -1.0 0.5 2 ::: {-220..-201} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.5 -1.0 0.5 2 ::: {-240..-221} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.0  1.0 0.5 2 ::: {-260..-241} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 0.5  1.0 0.5 2 ::: {-280..-261} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode M 1000 1.0  1.0 0.5 2 ::: {-300..-281} &

#Females
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.0 -1.0 0.5 0 ::: {-120..-101} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.5 -1.0 0.5 0 ::: {-140..-121} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.0  1.0 0.5 0 ::: {-160..-141} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.5  1.0 0.5 0 ::: {-180..-161} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 1.0  1.0 0.5 0 ::: {-200..-181} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.0 -1.0 0.5 2 ::: {-220..-201} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.5 -1.0 0.5 2 ::: {-240..-221} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.0  1.0 0.5 2 ::: {-260..-241} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 0.5  1.0 0.5 2 ::: {-280..-261} &
parallel --max-procs 5 python3.12 FitSTD.py {1} 1 $mode F 1000 1.0  1.0 0.5 2 ::: {-300..-281} &

#castre
parallel --max-procs 4 python3.12 FitSTD.py {1} 1 $mode c 1000 0.0 -1.0 0.5 0 ::: {-120..-101} &
parallel --max-procs 4 python3.12 FitSTD.py {1} 1 $mode c 1000 0.5 -1.0 0.5 0 ::: {-140..-121} &
parallel --max-procs 4 python3.12 FitSTD.py {1} 1 $mode c 1000 0.0  1.0 0.5 0 ::: {-160..-141} &
parallel --max-procs 4 python3.12 FitSTD.py {1} 1 $mode c 1000 0.5  1.0 0.5 0 ::: {-180..-161} &
parallel --max-procs 4 python3.12 FitSTD.py {1} 1 $mode c 1000 1.0  1.0 0.5 0 ::: {-200..-181} &
parallel --max-procs 3 python3.12 FitSTD.py {1} 1 $mode c 1000 0.0 -1.0 0.5 2 ::: {-220..-201} &
parallel --max-procs 3 python3.12 FitSTD.py {1} 1 $mode c 1000 0.5 -1.0 0.5 2 ::: {-240..-221} &
parallel --max-procs 3 python3.12 FitSTD.py {1} 1 $mode c 1000 0.0  1.0 0.5 2 ::: {-260..-241} &
parallel --max-procs 3 python3.12 FitSTD.py {1} 1 $mode c 1000 0.5  1.0 0.5 2 ::: {-280..-261} &
parallel --max-procs 3 python3.12 FitSTD.py {1} 1 $mode c 1000 1.0  1.0 0.5 2 ::: {-300..-281} &

#ovx
parallel --max-procs 40 python3.12 FitSTD.py {1} 1 $mode ovx 1000 0.0 1.0 -1 0 -1 ::: {-120..-1} &
parallel --max-procs 40 python3.12 FitSTD.py {1} 1 $mode ovx 1000 0.0 1.0 -1 0 -1 ::: {-240..-121} &
parallel --max-procs 25 python3.12 FitSTD.py {1} 1 $mode ovx 1000 0.0 1.0 -1 4 -1 ::: {-360..-241} &
parallel --max-procs 120 python3.12 FitSTD.py {1} 1 $mode ovx 1000 0.0 1.0 -1 0 -1 ::: {0..15000} &

#Run bootstrapping
parallel --max-procs 50 python3.12 FitSTD.py {1} 1 $mode M 1000 0.0 1.0 0.5 0 1.0 ::: {0..12000} &
parallel --max-procs 50 python3.12 FitSTD.py {1} 1 $mode F 1000 0.0 1.0 0.5 0 1.0 ::: {0..12000} &
parallel --max-procs 35 python3.12 FitSTD.py {1} 1 $mode c 1000 0.0 1.0 0.5 0 1.0 ::: {0..12000} &


#Analyse simulations
python3.12 FitSTD.py -1000 1 0 M 1000 &
python3.12 FitSTD.py -1000 1 0 F 1000 &
python3.12 FitSTD.py -1000 1 0 c 1000 &


python3.12 FitSTD.py -2000 1 0 M 1000 &
python3.12 FitSTD.py -2000 1 0 F 1000 &
python3.12 FitSTD.py -2000 1 0 c 1000 &

python3.12 FitSTD.py -2000 1 0 ovx 1000 &
python3.12 FitSTD.py -2000 1 0 mcM 1000 &
python3.12 FitSTD.py -2000 1 0 mcF 1000 &

#Notch and Lgr6 
#Sweep
parallel --max-procs 30 python3.12 FitSTD.py {1} 1 $mode notchF 1000 0.0 1.0 -1 0 -1 ::: {0..15000} &
parallel --max-procs 30 python3.12 FitSTD.py {1} 1 $mode notchM 1000 0.0 1.0 -1 0 -1 ::: {0..15000} &
parallel --max-procs 30 python3.12 FitSTD.py {1} 1 $mode lgr6F 1000 0.0 1.0 -1 0 -1 ::: {0..15000} &
parallel --max-procs 30 python3.12 FitSTD.py {1} 1 $mode lgr6M 1000 0.0 1.0 -1 0 -1 ::: {0..15000} &

python3.12 FitSTD.py -1000 1 0 notchF 1000 &
python3.12 FitSTD.py -1000 1 0 notchM 1000 &
python3.12 FitSTD.py -1000 1 0 lgr6F 1000 &
python3.12 FitSTD.py -1000 1 0 lgr6M 1000 &
python3.12 FitSTD.py -1000 1 0 ovx 1000 &

# Simulated annealing - Notch and Lgr6
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode notchM 1000 0.0 1.0 -1 0 -1 ::: {-015..-001} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode notchM 1000 1.0 1.0 -1 0 -1 ::: {-030..-016} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode notchF 1000 0.0 1.0 -1 0 -1 ::: {-015..-001} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode notchF 1000 1.0 1.0 -1 0 -1 ::: {-030..-016} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode lgr6M 1000 0.0 1.0 -1 0 -1 ::: {-015..-001} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode lgr6M 1000 1.0 1.0 -1 0 -1 ::: {-030..-016} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode lgr6F 1000 0.0 1.0 -1 0 -1 ::: {-015..-001} &
parallel --max-procs 15 python3.12 FitSTD.py {1} 1 $mode lgr6F 1000 1.0 1.0 -1 0 -1 ::: {-030..-016} &


python3.12 FitSTD.py -2000 1 0 notchF 1000 &
python3.12 FitSTD.py -2000 1 0 notchM 1000 &
python3.12 FitSTD.py -2000 1 0 lgr6F 1000 &
python3.12 FitSTD.py -2000 1 0 lgr6M 1000 &

'''


import sys
import numpy as np

#If we pass in all parameters as a string
if len(sys.argv) == 2:
    sys.argv = [sys.argv[0]] + sys.argv[1].split()

i = int(sys.argv[1])


if len(sys.argv) > 2:
    ncores = int(sys.argv[2])
else:
    ncores = 1

if len(sys.argv) > 3:
    mode = int(sys.argv[3])
else:
    mode = 0

if len(sys.argv) > 4:
    sex = sys.argv[4]
else:
    sex = 'M'

if len(sys.argv) > 5:
    ncells = int(sys.argv[5])
else:
    ncells = 1024
    L = np.sqrt(ncells)

if len(sys.argv) > 6:
    cTD = float(sys.argv[6])
else:
    cTD = 0.0

if len(sys.argv) > 7:
    cST = float(sys.argv[7])
else:
    cST = 0.0

if len(sys.argv) > 8:
    cvS = float(sys.argv[8])
else:
    cvS = 0.5

if len(sys.argv) > 9:
    met_mode = int(sys.argv[9])
else:
    met_mode = 0

if len(sys.argv) > 10:
    cvD = float(sys.argv[10])
else:
    cvD = -1

if len(sys.argv) > 11:
    tau = float(sys.argv[11])
else:
    tau = 100.0

if len(sys.argv) > 12:
    hilln = float(sys.argv[12])
else:
    hilln = 16.0

start_from_best = True

tscale_inc_times = np.arange(2,6)
T = -30


# Set rho based on MF value and set initial parameters for q and r for SA
if sex == 'M':
    exp_file = '../experiment/homeo/data_ctr_M.pkl'
    rhoS = 0.3
    cv_sb = [1.0]
    q = 1.0
    r = 0.2
elif sex == 'F':
    exp_file = '../experiment/homeo/data_ctr_F.pkl'
    rhoS = 0.275
    cv_sb = [0.1]
    q = 1.0
    r = 0.2
elif sex == 'c':
    exp_file = '../experiment/homeo/data_castre.pkl'
    rhoS = 0.275
    cv_sb = [1.0]
    q = 1.0
    r = 0.2
elif sex == 'ovx': 
    exp_file = '../experiment/ovx/data_Ovx.pkl'
    rhoS = 0.32
    cv_sb = np.arange(0.1, 1.1, 0.15) #[1.0]
    q = 1.0
    r = 0.2
    tscale_inc_times = np.arange(1,5)
elif sex == 'notchF':
    exp_file = '../experiment/notch/data_Notch1_F.pkl'
    rhoS = 0.26
    cv_sb = np.arange(0.1, 1.1, 0.15) #[1.0]
    q = 1.0
    r = 0.2
    tscale_inc_times = np.arange(1,5)
    T = -15
elif sex == 'notchM':
    exp_file = '../experiment/notch/data_Notch1_M.pkl'
    rhoS = 0.34
    cv_sb = np.arange(0.1, 1.1, 0.15) #[1.0]
    q = 1
    r = 0.2
    T = -15
    tscale_inc_times = np.arange(1,5)
elif sex == 'lgr6M':
    exp_file = '../experiment/lgr6/data_Lgr6_M.pkl'
    rhoS = 0.26 #0.26
    cv_sb = np.arange(0.1, 1.1, 0.15) #[1.0]
    q = 1.0
    r = 0.2
    T = -15
    tscale_inc_times = np.arange(1,5)
elif sex == 'lgr6F':
    exp_file = '../experiment/lgr6/data_Lgr6_F.pkl'
    rhoS = 0.26
    cv_sb = np.arange(0.1, 1.1, 0.15) #[1.0]
    q = 1.0
    r = 0.2
    T = -15
    tscale_inc_times = np.arange(1,5)
else:
    raise ValueError("Invalid sex specified")


#Imports
import matplotlib.pyplot as plt

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




#Metrics
ksdists  = {'dist': ['S2like.basalCS', 'S2like.supraCS'], 'dist_metric': 'ksd', 'name': 'ksd_b_sb'}
k1dists  = {'dist': ['S2like.basalCS'], 'dist_metric': 'ksd', 'name': 'ksd_bas'}
lldistS  = {'dist': ['S4like.basalCS'], 'dist_metric': 'lLL', 'name': 'll_S_bas'}
ll_dist_STD = {'dist': ['Slike.fullbasalCS', 'Slike.fullsupraCS',
                    'Tlike.fullbasalCS', 'Tlike.fullsupraCS',
                    'Dlike.fullbasalCS', 'Dlike.fullsupraCS', 
                    'S2like.xfrac'], 
                    'dist_metric': 'lLL', 'name': 'll_STD'}

ll_dist_SPD = {'dist': ['S4like.fullbasalCS', 'S4like.fullsupraCS',
                    'CPlike.fullbasalCS', 'CPlike.fullsupraCS',
                    'Dlike.fullbasalCS', 'Dlike.fullsupraCS'], 
                    'dist_metric': 'lLL', 'name': 'll_SPD'}
full = {'custom_metrics': {'S4like.fulljointCS': cmet.logLL_joint, 
                           'CPlike.fulljointCS': cmet.logLL_joint, 
                           'Dlike.fulljointCS': cmet.logLL_joint}, 
                           'name': 'full'}



if met_mode == 0:
    T0 = 50
    mets = [ll_dist_SPD, ksdists] # [ll_dist_SPD, ll_dist_STD, ksdists, k1dists, lldistS, full] 
elif met_mode == 1:
    T0 = 50
    mets = [ll_dist_STD, ll_dist_SPD, ksdists, k1dists, lldistS]
elif met_mode == 2:
    T0 = 0.0275
    mets = [ksdists, ll_dist_SPD, ll_dist_STD, k1dists, lldistS]
elif met_mode == 3:
    T0 = 5
    mets = [lldistS, ksdists, ll_dist_SPD, ll_dist_STD, k1dists]
elif met_mode == 4:
    T0 = 1
    mets = [full, lldistS, ksdists, ll_dist_SPD, ll_dist_STD, k1dists]



compute_kwargs = {'Dlike': {'containsBasal': False}}
toCalculate = ['basalCS']


extra_params = {}

if i < 0:
    freeParams = { 'r': [0.05,0.35], 
                  'corrDD': [0.0,1.0],  
                  'q': [0.4,1.0], 
                  'CVd_S': [0.2, 1.0],
                  'CVd_D': [0.2, 1.0]}    
    extra_params['corrSS'] = 0.5
else:
    freeParams = { 'r': np.arange(0.05,0.51,0.05), 
                  'corrSS': np.arange(0.1,1.01,0.1), 
                  'corrDD': np.arange(0.1,1.01,0.1),  
                  'q': np.arange(0.4,1.01,0.05)}


if cvD > 0:
    extra_params['CVd_D'] = cvD
else:
    freeParams['CVd_D'] = [1.0]

if cvS > 0:
    extra_params['CVd_S'] = cvS
else:
    freeParams['CVd_S'] = [1.0]



if mode == 0:
    if i >= 0:
        savefolder = 'ST2D_0D_finale_boots/'
    else:
        savefolder = 'ST2D_0D_finale/'
    nseeds = 1
    params = {'tau': tau, 'refractT': -1, 'corrTT': -1.0,
                'CVd_T': -1.0, 'corrTD': cTD, 'rhoS': rhoS,
                'corrST':  cST}
    s0 = topt.sim_anneal(sim_model.ST2D_0D, freeParams,
                        params = {**params, **extra_params}, simtype = 'h',
                    exp_file = exp_file,  exp_prefix = 'ctr', tshift = 2.0, metrics = mets,
                    compute_kwargs = compute_kwargs, toCalculate = toCalculate,
                move_seed = np.abs(i),  use_linear = ['r', 'rhoS', 'q', 'CVd_S', 'CVd_D', 'corrST', 'corrDD', 'corrSS'],
                move_sizes =     {'r': 0.025, 'rhoS':  0.025, 'q': 0.025, 'CVd_S':  0.05, 'CVd_D':  0.05, 'corrST':  0.1, 'corrDD':  0.1, 'corrSS':  0.1},
                free_param_min = {'r': 0.00, 'rhoS':   0.1, 'q':  0.0, 'CVd_S':  0.0, 'CVd_D':  0.0, 'corrST':  0.0, 'corrDD':  0.0, 'corrSS':  0.0},
                free_param_max = {'r': 1.00, 'rhoS':   1.0, 'q':  1.01, 'CVd_S': 1.04, 'CVd_D': 1.04, 'corrST': 1.04, 'corrDD': 1.04, 'corrSS': 1.04},
                use_momentum = True)
elif mode == 1:
    savefolder = 'ST2D_2D_finale/'
    nseeds = ncores
    params = {'tau': tau, 'refractT': -1, 'corrTT': -1.0,
                    'CVd_T': -1.0, #'rhoS': rhoS,
                    'corrST':  cST, 'corrTD': cTD, 
                    'L': L, 'hilln': hilln}
    s0 = topt.sim_anneal(sim_model.ST2D_size, freeParams,
                        params = {**params, **extra_params}, simtype = 'h',
                    exp_file = exp_file,  exp_prefix = 'ctr', tshift = 2.0, metrics = mets,
                    compute_kwargs = compute_kwargs, toCalculate = toCalculate,
                move_seed = np.abs(i),  use_linear = ['r', 'rhoS', 'q', 'CVd_S', 'CVd_D', 'corrST', 'corrDD', 'corrSS'],
                move_sizes =     {'r': 0.025, 'rhoS':  0.025, 'q': 0.025, 'CVd_S':  0.05, 'CVd_D':  0.05, 'corrST':  0.1, 'corrDD':  0.1, 'corrSS':  0.1},
                free_param_min = {'r': 0.00, 'rhoS':   0.1, 'q':  0.0, 'CVd_S':  0.0, 'CVd_D':  0.0, 'corrST':  0.0, 'corrDD':  0.0, 'corrSS':  0.0},
                free_param_max = {'r': 1.00, 'rhoS':   1.0, 'q':  1.01, 'CVd_S': 1.04, 'CVd_D': 1.04, 'corrST': 1.04, 'corrDD': 1.04, 'corrSS': 1.04},
                use_momentum = False)
        


if i >= 0:
    print(s0.npsets)
    s0.optimise(ncells = ncells, tsteps = T, seeds = np.arange(1), savefolder = savefolder,
        tscalevals = -np.arange(2.0,12,0.05), shedvals = -np.arange(0.45, 1.1, 0.05), 
                postopt_params = {'sbas_cv': cv_sb}, 
        use_saved_metrics = False, ncores = 0, save_every = 5, merge_during = i < 0,
        mergefirst = False, init_params = {'relaxsteps': 0, 'homeosteps': 1000}, sim_args = {},
        skip = 5, save_sweeps = False, par_seed = True, i = i*(i > 0) + -1*(i < 0), nboots = 100, save_to_runspace = False,
        postopt_settings = {'tscale_by_quantity': 'S2like.basalCS', 'tscale_inc_times': tscale_inc_times})
elif i == -1000:
    savefolder = 'ST2D_0D_finale_boots/'
    s0.merge_indexed_searchspace(np.arange(20000), ncells = ncells, tsteps = T, savefolder = savefolder)

if i <= -1 and i > -1000:
    if start_from_best:
        print('Starting from best')
        start_idx = None
    else:
        start_idx = np.abs(i) % s0.npsets

    s0.runSA(ncells = ncells, tsteps = T, seeds = np.arange(nseeds), savefolder = savefolder,
                tscalevals = -np.arange(2.0,12,0.05), shedvals = -np.arange(0.45, 1.1, 0.05), 
                postopt_params = {'sbas_cv': cv_sb}, merge_during = True,
                use_saved_metrics = True, ncores = ncores, save_every = 1, mergefirst = True,
                init_params = {'relaxsteps': 0, 'homeosteps': 1000}, sim_args = {'p0': 3.5, 'dt': 1e-2},
                SAsteps = 500, T0 = T0, start_idx = start_idx, skip = 5, save_sweeps = False,
                postopt_settings = {'tscale_by_quantity': 'S2like.basalCS', 'tscale_inc_times': tscale_inc_times})

if i < -1000:
    s0.merge_seeded_searchspace(move_seeds = np.arange(1000), ncells = ncells, tsteps = T, savefolder = savefolder)    
