import random
import tiss_optimize as topt
import numpy as np
import time
from joblib import Parallel, delayed
import utils

  
class sim_anneal(topt.optimizer):
    def __init__(self, model, freeParams, exp_file, params = None, metrics = None, #metrics
                 simtype = '', mergefirst = True,
                 move_seed = None, exp_prefix = '', exp_suffix = '', tshift = 0.0,
                 use_int = [], dp_round = 8, use_linear = [], free_param_min = None, 
                 free_param_max = None, move_sizes = None, use_momentum = True, **kwargs):
        
        topt.optimizer.__init__(self, model, freeParams, exp_file, params, metrics, #metrics
                 simtype, mergefirst, move_seed, exp_prefix, exp_suffix, tshift, use_int = use_int,
                 dp_round = dp_round, **kwargs)
        
        #Storing details and move details
        self.setup_move_details()        
        self._current_idx = 0 # points to index of parameters
        self._last_idx    = 0 # points to the index of the previous set of parameters
        
        #Set-up parameters outlining moves
        self.move_sizes = self.setup_param_dict(move_sizes, 1.1)
        self.free_param_min = self.setup_param_dict(free_param_min, 0.0)
        self.free_param_max = self.setup_param_dict(free_param_max, np.inf)        
        self.use_linear = self.setup_param_dict(use_linear, False, True)
        self.use_momentum = use_momentum
        
        #Set-up simulated annealing parameters
        self.T0 = 1
        self.temps = np.array([])
        
        #Fitting metric
        self.fit_metric = self.metrics[0].name
        
            
        
    @property
    def current_idx(self):
        self._current_idx = self.visited[-1]
        return self._current_idx
        
    @property
    def last_idx(self):
        if len(self.visited) >= 2:
            self._last_idx = self.visited[-2]
        else:
            self._last_idx = self.current_idx
        return self._last_idx
    
    def lastn_idx(self, n):
        if len(self.visited) >= (n + 1):
            return self.visited[-(n+1)]
        else:
            return self.lastn_idx(n - 1)
        
        
    def setup_move_details(self):
        '''
            Stores moves to understand parameter space explored by simulated annealing algorithm
        '''
        self.visited = [] #order of which parameters are visited
        self.rejected  = [] #stores rejected move in form (a,b)
        self.accepted  = [] #stores acceptd move in form (a,b)
                
        
    def set_thermometer(self, T0, SAsteps, SA_scale, relax_rule = 'linear'):
        '''
            Set temperature for simulations
        '''
        
        if relax_rule == 'linear':
            self.temps = Tlinear(SAsteps + 1, T0, SA_scale)
        elif relax_rule == 'geometric':
            self.temps = Tlinear(SAsteps + 1, T0, SA_scale)
        elif relax_rule == 'custom':
            self.temps = np.array(T0) #input T0 as temperature
            assert(len(self.temps) > SAsteps)
        else:
            self.temps = T0*np.ones(SAsteps) #input T0 as temperature


    def costs(self, row = slice(None, None, None)):
        return self.searchspace.iloc[row][self.fit_metric]
        
    
    def paramChange(self):
        ''' 
            Find the parameter change between the most recent and second most recent move
        '''
        return self.searchspace.iloc[self.current_idx][self.freeParamVars] - self.searchspace.iloc[self.last_idx][self.freeParamVars]

        
        
    
    def runSA(self, tsteps, ncells, seeds = 0, SAsteps = 100, start_idx = None, T0 = 1, SA_scale = None, relax_rule = 'linear', fit_metric = None,
              savefolder = '.', shedvals   = None, shed_opt = 'grid',
                 tscalevals = None, tsc_opt  = 'grid', mergefirst = None, 
                 use_saved_metrics = False, merge_during = False, save_every = 10, 
                sim_args = None, init_params = None, parallel = None, ncores = 1,
                save_full = True, save_to_runspace = False, skip = 10, 
                save_sweeps = False, nboots = 0, boot_exp = True, boot_seed = 0,
                **kwargs):
        '''
            Runs simulated annealing algorithm
            
            start_idx = -1, starts most recent from queue, start_idx = None picks the best fit
        '''
        
        if sim_args is None:
            sim_args = {}
            
        if init_params is None:
            init_params = {}
        
        #Run in parallel:
        if ncores > 1 and parallel is None:
            print('Running SA in parallel')            
            with Parallel(n_jobs = ncores) as parallel:
                self.runSA(tsteps, ncells, seeds, SAsteps, start_idx, T0, SA_scale, relax_rule, fit_metric,
                    savefolder, shedvals, shed_opt,
                    tscalevals, tsc_opt, mergefirst, 
                    use_saved_metrics, merge_during, save_every, 
                    sim_args, init_params, parallel = parallel, save_full = save_full,  
                    save_to_runspace = save_to_runspace, skip = skip, 
                    save_sweeps = save_sweeps, nboots = nboots, boot_exp = boot_exp, 
                    boot_seed = boot_seed, **kwargs)
            
            return
        
        #Set-up saving and merging of optimiser, should load from unseeded file
        self.use_unseeded_ss_prefix()
        self.setup_optimiser(tsteps, ncells, savefolder, mergefirst, use_saved_metrics)
        
        #Set prefix for saving
        self.use_seeded_ss_prefix()
        self.setup_optimiser(tsteps, ncells, savefolder, mergefirst, use_saved_metrics)
        
        #Set-up post-processing optimisers
        self.setup_postopt(tscalevals, shedvals, tsc_opt, shed_opt, save_sweeps,
                           nboots, boot_exp, boot_seed, tsteps, ncells, savefolder, seeds, **kwargs)
        
        #Set-up move details
        self.setup_move_details()        
        
        #Starting index for simulated annealing
        if start_idx is None or np.abs(start_idx) >= self.npsets:
            print('Starting from best fit')
            start_idx = self.best_fit(self.fit_metric, return_idx = True)
        elif start_idx < 0:
            start_idx = self.npsets - 1
            assert(start_idx >= 0)        
        self.visited.append(start_idx)
        assert(self.current_idx == start_idx)
        #self.current_idx = start_idx
        
        if save_every is None:
            merge_during = False        

        #Set up metric for simulated anneling
        if fit_metric is None:
            fit_metric = self.metrics[0].name
        assert(fit_metric in self.metric_names)
        self.fit_metric = fit_metric
        
        #Set-up temperature
        self.set_thermometer(T0, SAsteps, SA_scale, relax_rule)
        
        
        # Run simulated annealing        
        print('Starting at:', time.asctime())
        print('Simulated annealing over:', self.freeParamVars)
        print('Optimising parameters, init params:',  self.paramset(self.current_idx), 'seed:', str(self.move_seed))
        print('Min params:', self.free_param_min)
        print('Max params:', self.free_param_max)
        print('Move sizes:', self.move_sizes)        
        
        #Running first time-step
        self.run_sim(tsteps, ncells, self.current_idx, seeds, savefolder, mergefirst = False, merge_during = False, 
                        sim_args = sim_args, init_params = init_params, parallel = parallel, end_save = False, 
                        skip = skip, **kwargs)

        t = 1
        lastrejected = False
        known_visits = 0 #stores the number of visits to known points in a row
        nresets = 0 
        nfails = 0 #number of failed simulations
        i = 0
        

        while t < SAsteps + 1:
            print('************\n')            
            print('Step', t, 'temperature', topt.round_sf(self.temps[t],5))

            if i > 100*SAsteps: #upper limit on iterations
                print('Too many iterations, breaking')
                break

            i += 1
            
            upmove  = None
            pchoose = None
            
            if self.use_momentum and t > 1:
                paramchange = self.paramChange()
                
                #momentum move proposal if one parameter has changed and cost is improved
                if lastrejected is False and self.costs(self.current_idx) < self.costs(self.last_idx) and np.sum(paramchange != 0) == 1: 
                    upmove = int((np.sum(paramchange) > 0)*2 - 1)
                    pchoose = list(paramchange[paramchange != 0].index)[0]
                    print('Using a momentum move, up', upmove > 0, ', pchoose', pchoose)
 
                
            params = self.proposeconfig(upmove, pchoose)
            
            print('Proposed free params:', params.to_dict())
            
            #ensures metric is calculated correctly:
            param_row = self.getParamRow(params)      
            
            #Run simulation or depending on whether we've visited before
            if param_row is not None and param_row in self.visited and self.costs(param_row) < self.ss_defaults[self.fit_metric] and np.isnan(self.costs(param_row)) == False:
                #we don't need to run
                print('Already visited row', param_row)
                known_visits += 1
                if known_visits > 3*self.nfree**2:
                    if self.costs(self.current_idx) <= np.min(self.costs()):
                        print('Too many consecutive re-visits, breaking')
                        break
                    else:
                        if nresets > 3:
                            print('Too many resets, breaking')
                            break

                        print('Too many consecutive re-visits, resetting to minimum cost')
                        bestfit_p = self.best_fit(self.fit_metric, return_idx = True)
                        self.visited.append(bestfit_p)
                        known_visits = 0
                        nresets += 1
                else:
                    self.visited.append(param_row)                    
            else: #Will run again
                if param_row is None:
                    self.visited.append(self.npsets)
                    self.add_paramset(params)
                else:
                    self.visited.append(param_row)
                
                try: #Ensures paramaters aren't proposed outside of bounds
                    self.run_sim(tsteps, ncells, self.current_idx, seeds, savefolder, mergefirst = False, merge_during = False, 
                        sim_args = sim_args, init_params = init_params, parallel = parallel, end_save = False, 
                        skip = skip, **kwargs)
                    accept = self.acceptMove(t)
                    known_visits = 0 # reset known visits
                    nfails = 0 # reset failed simulations
                except AssertionError:
                    if nfails > 3*self.nfree**2:
                        print('Too many failed simulations, breaking')
                        break
                    print('Failed to run simulation, proposing new')
                    accept = False
                    nfails += 1

            if t > 0 and not accept:
                self.visited.append(self.last_idx) #go back to last index
                t -= 1 #Re-run again
                lastrejected = True    
            else:
                lastrejected = False
                
            if merge_during and ((t + 1) % save_every == 0) and (lastrejected is False):
                print('Saving checkpoint, t = ', t)
                self.save_searchspace(tsteps, ncells, savefolder)
                if save_full:
                    self.save_full_object(tsteps, ncells, savefolder)

            print('Cost:', self.costs(self.current_idx))    
            t += 1 #iterate
                

        print('Ending at cost:', np.min(self.costs()), 'SA steps:', t, 'rejected steps:', len(self.rejected), ' at', time.asctime())
        print('Best fit, min cost: ')
        print(self.regularise_params(self.best_fit(self.fit_metric).to_dict()))


        self.save_searchspace(tsteps, ncells, savefolder)
        if save_full:
            self.save_full_object(tsteps, ncells, savefolder)        
    
        if save_to_runspace is True:
            self.save_searchspace_to_runspace(tsteps, ncells, savefolder)    
    
    
    def proposeconfig(self, upmove = None, pchoose = None):
        '''
            Propose a comfiguration of paramters to use for the next simulation
                keeps proposing until we find a valid one
        '''
        
        params = self.freeparamset(self.current_idx).copy()
        
        if pchoose is None:
            pchoose = self.freeParamVars[random.randint(0,self.nfree - 1)] #choice
        
        if upmove is None:
            upmove  = 2*(random.randint(0,1) - 0.5)
                        
        if self.use_linear[pchoose]:
            proposed = params[pchoose]  + self.move_sizes[pchoose]*upmove
        else:
            proposed = params[pchoose]*(self.move_sizes[pchoose]**upmove)
        
        #print('Changing', pchoose, 'Old:', params[pchoose], 'Proposed:', proposed)  
                
        #update parameter to proposed
        params[pchoose] = proposed
        params = self.regularise_params(params)

        repeat_last = (params == self.freeparamset(self.last_idx)).all() and (self.last_idx != self.current_idx)

        if proposed < self.free_param_min[pchoose] or proposed > self.free_param_max[pchoose] or repeat_last == True:
            print('Invalid or instant repeat, proposing new', params)
            params = self.proposeconfig()
        
        
        
        return self.regularise_params(params)



    def acceptMove(self, t, verbose = True):   
        '''
            Determines whe
        '''  
        accept = True
        temp = self.temps[t]
        dcost = self.costs(self.current_idx) - self.costs(self.last_idx)
        
        if temp == 0:
            acceptprob = 0.0
        else:
            acceptprob = np.exp(-dcost/temp)
        
        if acceptprob > 1.0:
            acceptprob = 1.0 
        
        if dcost > 0 and random.random() > acceptprob:
            accept = False
            
        #store moves
        if accept:
            self.accepted.append((self.last_idx, self.current_idx))
        else:
            self.rejected.append((self.last_idx, self.current_idx))            
        
        if accept and verbose:
            print('Accepted, Cost:', "{:.4g}".format(self.costs(self.last_idx)), '->', "{:.4g}".format(self.costs(self.current_idx)), 'at temp', "{:.4g}".format(temp), ', accept prob:', "{:.4g}".format(acceptprob))
        else:
            print('Rejected, Cost:', "{:.4g}".format(self.costs(self.last_idx)), '->', "{:.4g}".format(self.costs(self.current_idx)), 'at temp', "{:.4g}".format(temp), ', accept prob:', "{:.4g}".format(acceptprob))
            
        return accept  



def Tlinear(SAsteps, T0, SA_scale = 1.0):
    if SA_scale is None:
        SA_scale = 1.0
    return utils.relu(T0*(SAsteps - np.arange(SAsteps)*SA_scale)/SAsteps)

def Tgeom(SAsteps, T0, SA_scale = 0.5):
    if SA_scale is None:
        SA_scale = 0.5    
    return T0*SA_scale**np.arange(SAsteps)