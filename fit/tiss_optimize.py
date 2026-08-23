import numpy as np
import pandas as pd
import inspect
import sim_model
import os
import metrics
import datasets as ds
import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import glob
import calc_order_params as cop
import tissue
from itertools import product
import copy




class simulator():
    '''
        simulator class:
            The default behaviour is to just run simulations over multiple sets of parameter], that can be used as as a base for a brute force optimiser
            The sub-classes: optimizer, sim_anneal can fit simulations to particular metrics
            
        The set of parameters and features of parameters explored is a searchspace (represented 
            as a pandas df). The parameter sets we have explored is called (without additonal details)
            is the sub-dataframe fullparams

    '''
    
    def __init__(self, model, freeParams, params = None, move_seed = None, simtype = '', mergefirst = True,
                 use_int = [], dp_round = 8, toCalculate = None, order_params = None, analyse_extinct = False,
                 compute_kwargs = None, save_suffix = ''):
        
        '''
            How to input parameters?
            
                  model:  input the sim_model function to create the tissue class, e.g. sim_model.S_0D
             freeParams: either input as a dictionary, dictionary of lists, or list of dictionaries, or a tuple of (tuple param names, np array of values)
                            a)    specify single param set: dictionary             e.g. {'r': 0.2, 'tau': 100}
                            b)  grid of param combinations: dictionary of lists    e.g. {'r': [0.2, 0.3], 'tau': [100, 200]}, runs for (r,tau) = (0.2,100),(0.3,100),(0.2,200),(0.3,200)
                            c) multiple param combinations: list of dictionaries   e.g. [{'r': 0.2, 'tau': 100}, {'r': 0.2, 'tau': 200}] 
                            d) multiple param combinations: tuple of (tuple,array) e.g. (('r','tau'), np.array([[0.1,10], [0.2,20], [0.3, 20]])) 
                                    if 1D: elements correspond to parameter values
                                    if 2D (mxn): the m rows correspond to different parameter sets (of n parameters)
                 params: dictionary for the fixed parameters that are not left to their default values
                options: fitting options
                expfile: address of the pickle file where the experimental data is saved,
                            assumes that the data contains the quantities that we will compare to data
                            
                simtype: either 'h': homeostatic / 'm': mutated / '': regular simulation / None: does not change
        
                mergefirst
                
                use_int, specifies which parameters are integers
               dp_round, specifies how many significant figures to round to for floats
                    using None, does no rounding
                    
                toCalculate: list of quantities to calculate from metrics
                order_params: list of order parameters to calculate from metrics and save in search-space

                compute_kwargs: dict of dict of to specify kwargs for computing specific quantities
        '''

        #Set-up seed for proposing simulations, if used
        self.move_seed = move_seed
        if move_seed is not None:
            random.seed(int(move_seed))
        
        #Specify model and simulation type
        self.setup_model(model)
        self.save_prefix = ''  
        self.save_suffix = save_suffix      
        self.set_simtype(simtype)        
        
        #Order parameters defined
        if order_params is None:
            order_params = []
        
        self.order_params = order_params        
        
        #Set-up parameters and search-space
        self.setup_searchspace(freeParams, params)
        self.setup_regularise(use_int = use_int, dp_round = dp_round)        
        self.mergefirst = mergefirst #merges parameters on first run if existing runspace exists               
        
        #Pre-fixes for consistent saving
        self.searchspace_prefix = 'runspace'
        
        #Get quantities to calculate, if empty we won't calculate anything
        self.toCalculate = self.getToCalculate(toCalculate = toCalculate, order_params = order_params)
        self.analyse_extinct = analyse_extinct #whether to calculate order parameters and metrics for extinct sims

        if compute_kwargs is None:
            self.compute_kwargs = {}
        else:
            self.compute_kwargs = compute_kwargs


    def copy(self):
        return copy.deepcopy(self)

    def use_unseeded_ss_prefix(self):
        '''
            Revert to standard prefixing method
        '''
                    
        self.searchspace_prefix = 'runspace'


    def use_seeded_ss_prefix(self, s):
        '''
            Naming convention for multiple parallel runs of fitting algorithm
        '''
                     
        self.searchspace_prefix = 'runspace_seed_' + str(s)


    def use_indexed_ss_prefix(self, i):
        '''
            Naming convention for multiple parallel runs of fitting algorithm
        '''
                       
        self.searchspace_prefix = 'runspace_index_' + str(i)

    def save_full_object(self, tsteps, ncells, savefolder = '.', obj_prefix = None):
        '''
            Saves full object for use in analysis
        '''
        if obj_prefix is None:
            obj_prefix = 'full' + self.searchspace_prefix
        sn = self.searchspace_save_name(tsteps, ncells, ss_prefix = obj_prefix)
        ds.save_object(self, savefolder + '/' + sn)
        return self
    

    def setup_model(self, model):
        '''
            Sets-up model and checks we are in basic list of models
        '''
        self.model = model
        self.modelname = model.__name__
        if self.modelname in sim_model.models0D:
             self.model_is_2D = False
        elif self.modelname in sim_model.models2D:
            self.model_is_2D = True
        else:
            print('Invalid model')
            assert(0)

        
    def set_simtype(self, simtype):
        if simtype == 'h':
            self.use_h_prefix()
            self.sim = self.hsim
        elif simtype == 'm':
            self.use_m_prefix()
            self.sim = self.msim
        elif simtype == 'a':
            self.use_a_prefix()
            self.sim = self.asim
        elif simtype is not None: 
            self.reset_prefix()
            self.sim = self.rsim

    def npops(self):
        if self.sim == self.msim:
            return sim_model.npops(self.model) + 1
        else:
            return sim_model.npops(self.model)
        

    def use_h_prefix(self): #pre-fix for saving a homeostatic simulation
        self.save_prefix = 'h'
        return self.save_prefix
    
    def use_a_prefix(self): #pre-fix for saving a ablation simulation
        self.save_prefix = 'a'
        return self.save_prefix
    
    def use_m_prefix(self): #pre-fix for saving a mutated simulation
        self.save_prefix = 'm'
        return self.save_prefix
    
    def reset_prefix(self):
        self.save_prefix = ''
        return self.save_prefix
    
    def check_params(self):
        #Check that all the defined "parameters", actually exist
        poss = self.possible_params()
        for p in list(self.fullparams.columns):
            assert p in poss , 'Parameter ' + p + ' not in possible parameters'
    
    def possible_params(self):
        if self.model_is_2D:
            tfn = sim_model.tissue.simFull
        else:
            tfn = sim_model.tissue0D.simFull
        
        return list(inspect.signature(self.model).parameters) + list(inspect.signature(tfn).parameters) + list(inspect.signature(self.sim).parameters)
    
    
    def set_to_extinct(self, i):
        '''
            Sets flag to indicate extinction
        '''

        if 'extinct' not in self.searchspace.columns:
            self.searchspace['extinct'] = False

        self.searchspace.at[i, 'extinct'] = True
    
    def regularise_params(self, params):
        '''
            Takes the dictionary use_int and turns parameter set into the right form
                and the parameters that are not integers round to a certain number of
                significant figures, which by default is 5
        '''
        
        if isinstance(params, pd.DataFrame):
            #print('Regularising dataframe')
            columns = params.columns
        else:
            columns = params.keys()
        
        for p in columns:
            if p in self.use_int:
                if self.use_int[p]:
                    params[p] = convert_to_int(params[p]) #works from df, numpy array or just number
                else:
                    if self.dp_round is not None:
                        params[p] = np.round(params[p], self.dp_round) #round_sf(params[p], self.dp_round)
                
        return params
        
        
    def setup_regularise(self, use_int = [], dp_round = None, regularise = True):
        '''
            Sets up dictionary on how to regularise paramaters:
                see self.regularise_params
        '''
        self.use_int  = self.setup_param_dict(use_int, False, True)  
        self.dp_round = dp_round
        
        if regularise:
            #Regularise the existing searchspace, this is new parameters not already saved ones
            self.regularise_params(self.searchspace)
        
    
 
        
    def setup_param_dict(self, d1, default = True, inlistval = None):
        '''
            Sets up param dictionary with default values if any free 
                parameters are missing
                
            can input either a list of keys or a dictionary
            
            if list we set those in d1 to inlistval, and others to default
        '''
        
        if d1 is None:
            d1 = {}
        
        d2 = {}
        if type(d1) is list:
            if inlistval is None:
                inlistval = not default
            d2 = {k: inlistval for k in d1}
        elif type(d1) is dict:
            d2 = d1
        else:
            print('Invalid type provided')
            assert(0)
            
        for pname in self.freeParamVars:
            if pname not in d2.keys():
                d2[pname] = default            
                       
        return d2    
    
    def update_param_seed(self, i, seed):
        self.searchspace.at[i,'maxseed'] = seed
    
    
    def setup_searchspace(self, freeParams, params):
        if params is None:
            params = {}
        
        self.setup_params(freeParams, params)        
        self.check_params()
        self.searchspace = self.searchspace.reset_index() #add an index
        
        #Keep track of the maximum seed we have run simulation for    
        #Assume that all seeds from 0 to this have been run, for purpose of sims
        
        #Set-up defaults
        self.set_ss_defaults()
        self.update_to_defaults() #update categories to defaults
        
        #Order parameters
        if self.order_params is not None:
            for op in cop.op_names(self.order_params):
                self.searchspace[op] = None #np.nan        
        
        
    def add_paramset(self, paramset):
        '''
            Add a parameter set and returns the full search-space
        '''
        if not self.on_runlist(paramset):
            rowidx = self.npsets            
            fullpset = {**self.ss_defaults, **self.fixedParams, **paramset, 'index': rowidx}
            newrow = pd.DataFrame([list(fullpset.values())], columns = list(fullpset.keys()))
            self.searchspace = pd.concat([self.searchspace, newrow], ignore_index=True)
        
        return self.searchspace
        
    def set_ss_defaults(self):
        self.ss_defaults = {}
        self.ss_defaults['maxseed'] = -1

        
    
    def update_to_defaults(self):
        for c in self.ss_defaults.keys():
            if c not in self.searchspace.columns:
                self.searchspace[c] = self.ss_defaults[c]
        self.ss_nans_to_defaults()
        
        
    def ss_nans_to_defaults(self):
        '''
            Sets all the nan values in the search-space to the default values
        
        '''
        self.searchspace.fillna(self.ss_defaults, inplace=True)
        
    def remove_parameter_duplicates(self):
        self.searchspace = self.searchspace.drop_duplicates(subset = self.allParamVars)
        self.searchspace = self.searchspace.reset_index(drop = True) # drop = True
        
    
    def setup_params(self, freeParams, params):
        '''
        #Process freeParams, convert from input form to (tuple, array) form
        #We extracts the following
        #freeParamVars: name of the variables of free parameters
        #freeParamVals: values of free parameters 
        #Assumes parameters are a single number or boolean
        '''
        
        self.fixedParams = params
        
        if   type(freeParams) == dict:
            self.freeParamVars = list(freeParams.keys())
            self.nfree = len(self.freeParamVars)
            freeParamVals = np.array(np.meshgrid(*freeParams.values())).T.reshape(-1, self.nfree)
            
        elif type(freeParams) ==  list:
            self.freeParamVars = list(freeParams[0].keys())
            self.nfree = len(self.freeParamVars)
            freeParamVals = np.zeros((0, self.nfree))
            for parset in freeParams: #add valid parameter sets
                #assert(len(parset) == self.nfree)
                if len(parset) == self.nfree:
                    freeParamVals = np.vstack([freeParamVals, np.array([parset[var] for var in self.freeParamVars])])
                
        elif type(freeParams) == tuple:
            #Check the format is valid
            assert(len(freeParams) == 2)
            self.nfree = len(freeParams[0])  
            self.freeParamVars = list(freeParams[0])
            freeParamVals = np.reshape(freeParams[1], (-1, self.nfree))
            assert(len(freeParamVals[0,:]) == self.nfree)
                      
        #Ensure no overlapping parameters:
        for p in self.freeParamVars:
            if p in self.fixedParams:
                self.fixedParams.pop(p)

        #Panda dataframe for complete set of specified parameters defined by the model
        self.searchspace = pd.DataFrame(freeParamVals, columns = self.freeParamVars)
        for p in params.keys():
            self.searchspace[p] = params[p]
        
        #Number of parameter sets
        #self.npsets = self.fullparams.shape[0] #set with property
        self.allParamVars = list(self.searchspace.columns)
        self.allParamVars.sort()
        self.remove_parameter_duplicates()
        
    
    def addparams(self, params):
        '''
            Add parameters to the searchspace
        '''
        for p in params.keys():
            if p not in self.allParamVars:
                self.allParamVars.append(p)
                self.searchspace[p] = params[p]
                self.fixedParams[p] = params[p]
            else:
                print('Warning: parameter already exists in searchspace:', p)
                self.searchspace[p] = params[p]
                
        #self.npsets = self.fullparams.shape[0] #set with property
        self.remove_parameter_duplicates()
           
    
    @property
    def fullparams(self):
        return self.searchspace[self.allParamVars]
              
    @property
    def npsets(self):
        return self.fullparams.shape[0]
    
    
    def paramset(self, p):
        '''
            Return the p-th parameter set as a dictionary
        '''
        #stopped doing this as this changes datatype of ints, which leads to weird behaviour
        #return dict(self.fullparams.iloc[p]) 
        return {cn: self.fullparams.at[p, cn]  for cn in self.allParamVars}

    def freeparamset(self, p, usedict = False):
        '''
            Returns the free parameters of the p-th parameter set as a dictionary,
                else return a pandas series
        '''
        if usedict:
            return {cn: self.fullparams.at[p, cn]  for cn in self.freeParamVars}
                    
        else:
            return self.searchspace.iloc[p][self.freeParamVars]


    def sim_params(self, p):
        '''
            From parameters p-th param set, we extract those to pass into simulation
        '''
        pset = self.paramset(p)
        
        if self.model_is_2D:
            fn = sim_model.tissue.simFull
        else:
            fn = sim_model.tissue0D.simFull
        
        sp =  self.regularise_params({key: value for key, value in pset.items() if key in list(inspect.signature(fn).parameters)})
        if 'mu' in sp.keys():
            print('Warning: mu may be used as division rate and friction coefficient. Using mu = 1')
            sp.pop('mu') #remove mu from sim params
        return sp
    
        
    def model_params(self, p):
        '''
            From parameters p-th param set, we extract those to pass into model
        '''   
        pset = self.paramset(p)
        
        return self.regularise_params({key: value for key, value in pset.items() if key in list(inspect.signature(self.model).parameters)})
    

    def other_params(self, p):
        '''
            From parameters p-th param set, we extract those that are neither
                model of sim params            
        '''
        pset = self.paramset(p)        
        other_keys = list(set(pset.keys()) - set(self.sim_params[p].keys()) - set(self.model_params[p].keys()))
        return self.regularise_params({key: pset[key] for key in other_keys})
    
    
    def sim_save_glob(self, tsteps, ncells, seed = 0, savefolder = '.'):
        ''' 
            Generates glob for save name for simulations
            
            Need to ensure that the correct prefix is set to avoid unwanted behaviour            
            paramsdf dataframe defaults to self.fullparams
            
            If params = None
        '''
        
        savename = 'run_' + self.modelname + '_t_' + str(tsteps)
        
        if np.ndim(ncells) == 0:
            savename += '_n_' + str(ncells)
        else:
            savename += '_n'
            for i in range(len(ncells)):
                savename += '_' + str(ncells[i])
        
        for cn in self.allParamVars:
            savename +=  '_' + cn + '_*'
        
        if seed is not None:
            savename += '_seed_' + str(seed)

        if len(self.save_suffix) > 0:
            savename += '_' + self.save_suffix
        
        return glob.glob(savefolder + '/' + self.save_prefix + savename + '.pkl')

    
    def sim_save_name(self, tsteps, ncells, p = 0, seed = 0, params = None):
        ''' 
            Generates save name for simulations
            
            Need to ensure that the correct prefix is set to avoid unwanted behaviour            
            paramsdf dataframe defaults to self.fullparams
            
            If params = None
        '''
        
        if params is None:
            params = self.fullparams.iloc[p]
        
        savename = 'run_' + self.modelname + '_t_' + str(tsteps)
        
        if np.ndim(ncells) == 0:
            savename += '_n_' + str(ncells)
        else:
            savename += '_n'
            for i in range(len(ncells)):
                savename += '_' + str(ncells[i])
        
        for cn in self.allParamVars:
            savename +=  '_' + cn + '_' + str(params[cn])
        
        if seed is not None:
            savename += '_seed_' + str(seed)

        if len(self.save_suffix) > 0:
            savename += '_' + self.save_suffix    
        
        return self.save_prefix + savename + '.pkl'
    
    
    def searchspace_save_name(self, tsteps, ncells, ss_prefix = None):
        '''
            Save names for run space
        '''
        
        if ss_prefix is None:
            ss_prefix = self.searchspace_prefix
        
        savename = self.save_prefix + ss_prefix + '_' + self.modelname + '_t_' + str(tsteps)
        
        if np.ndim(ncells) == 0:
            savename += '_n_' + str(ncells)
        else:
            savename += '_n'
            for i in range(len(ncells)):
                savename += '_' + str(ncells[i])
        
        for cn in self.allParamVars:
            savename +=  '_' + cn

        if len(self.save_suffix) > 0:
            savename += '_' + self.save_suffix

        return savename + '.pkl'  
      
      
    def getToCalculate(self, toCalculate = None, order_params = None):
        '''
            Determine the quantities (including pdfs) to calculate from metrics
        '''
        
        if toCalculate is None:
            toCalculate = []
        
        toCalculate += cop.getToCalculate(order_params)
        
        return list(set(toCalculate))
    
    
    def test_homeostasis(self, ncells, tol = 1.5, dx = 200):
        '''
            Takes time-series of number of cells.
            
            Checks if changes from start to end are smaller than standard deviation across the series
            
            This is a bad test, but I need to think of a better one
            
        '''
        tsteps = len(ncells[:,0])
        if tsteps < dx:
            dx = tsteps
            
        change = np.max(np.abs(np.mean(ncells[:int(dx/2),:],0) - np.mean(ncells[-int(dx/2):,:],0))/np.std(ncells,0))
        
        return change < tol
    
    
    def init_process_sims(self, seeds, i = None, tsteps = None, skip = 10, tscalevals = 1, **kwargs):
        '''
            Initialise simulation data for processing
            
            tscalevals: can be used to set time rescaling
        '''
        
        if len(self.toCalculate) > 0:
            #print('t-scale: ', tscalevals)
            
            if i is None:
                params = {}
            else:
                params = self.paramset(i)
            sim_data_by_seed = ds.sim_dataset(self.toCalculate, {}, tsteps, seeds, npops = self.npops(), 
                                              tscalevals = tscalevals, skip = skip, sim_params = params, alltimes = True, 
                                              compute_kwargs = self.compute_kwargs, analyse_extinct = self.analyse_extinct)
            
            return sim_data_by_seed
            
        else:
            return ds.dataset()
        
    
    def process_sim_by_seed(self, sim_data_by_seed, results, seed = 0, **kwargs):
        '''
            This function allows processing of simulation, immediately after running
            In the future we can call an analysis script, or something. 
        '''
        if self.run_only or len(self.toCalculate) == 0:
            return sim_data_by_seed
        
        sim_data_by_seed = sim_data_by_seed.quantities_by_seed(results['tiss'], results, seed)

        return sim_data_by_seed        
            
    
    def process_sims(self, sim_data_by_seed, i, combineSims = True, **kwargs):
        '''
            This function allows processing of simulations, after collecting 
                individual data by seed
        '''
        
        if self.run_only or len(self.toCalculate) == 0:
            return sim_data_by_seed   
        
        if sim_data_by_seed.extinct:
            self.set_to_extinct(i)
            if not self.analyse_extinct:
                return
        
        if combineSims:
            sim_data = sim_data_by_seed.calculate_combined_quantities(full = False)
        else:
            sim_data = sim_data_by_seed
        
        self.store_order_params(sim_data, i)

    
    def store_order_params(self, sim_data, i):
        ''' 
            Given simulations we have analysed we store the order parameters
                to the search-space
        '''
        
        if len(self.order_params) == 0:
            return
        
        if not sim_data.collapsed:
            sim_data = sim_data.get_instance()
        
        allops = cop.ops_from_ds(sim_data, self.order_params)
        
        for op in allops:
            self.searchspace.at[i, op] = allops[op]
            

    
    def run_sim_by_seed(self, i, s, sim_data_by_seed, tsteps, ncells, 
                        savefolder, seedmanage = True, sim_args = None, init_params = None,
                        verbose = True, norun = False, skip = 10, shed_profile = None):
        '''
            Runs simulation for parameters "i" and seed "s" for simulation type
                "simtype" and initialisation parameters specified by 
                
            Saves data to "sim_data_by_seed"
            
            norun = True: stops if we don't load
            
            Use: shed_profile is not None, then we use a custom shed profile
        '''
        
        if sim_args is None:
            sim_args = {}
            
        if init_params is None:
            init_params = {}
        
        if seedmanage:
            if s == -1:
                s = i
            np.random.seed(actualseed(i,s))

        #Add skip to sim_args
        if 'skip' not in sim_args:
            sim_args['skip'] = skip
        elif sim_args['skip'] != skip:
            print('Warning skip defined twice, using:', sim_args['skip'])   
            
        if sim_args['skip'] < 0:
            tau = self.paramset(i)['tau']
            if tau > 0:
                sim_args['skip'] = int(np.abs(sim_args['skip']*tau))
            else:
                sim_args['skip'] = int(np.abs(sim_args['skip']*100)) #assume tau = 100 if not specified
            
            if sim_args['skip'] == 0:
                sim_args['skip'] = 1

        #Update maximum seed
        if seedmanage and s > self.searchspace.iloc[i]['maxseed']:
            self.update_param_seed(i, s)

        #Check if exists
        sn = self.sim_save_name(tsteps, ncells, i, s)                
        if os.path.exists(savefolder + '/' + sn):
            if verbose:
                print('File already exists:', sn)
            results = ds.load_object(savefolder + '/' + sn, deleteEOF = True)
        else:
            if norun:
                print('\n\nnot found:', sn)
                if 'missing' not in self.searchspace.columns:
                    self.searchspace['missing'] = False
                self.searchspace.at[i, 'missing'] = True
                return sim_data_by_seed
            
            #If negative tsteps we specify number of division times:
            tsteps1 = tsteps
            if tsteps1 < 0:
                tau = self.paramset(i)['tau']
                if tau > 0:
                    tsteps1 = int(np.abs(tsteps*tau))
                else:
                    tsteps1 = int(np.abs(tsteps*100)) #assume tau = 100 if not specified
            
            #Run simulations
            results = self.sim(tsteps1, ncells, i, sim_args = sim_args, **init_params)
            results['seed'] = actualseed(i,s)
            
            tissue.save_tissue(results, savefolder + '/' + sn)
            if verbose:
                print('Saving simulation to', sn)
            
        sim_data_by_seed = self.process_sim_by_seed(sim_data_by_seed, results, s) 
        
        #Using custom shed profile
        if shed_profile is not None:
            if s == 0:
                print('Using custom shed profile')
            sim_data_by_seed.rerun_suprabasal(results = results, tiss = results['tiss'], 
                                              sb = 0, seed = s, sbsi = 0, shed_profile = shed_profile)        
        
        return sim_data_by_seed
    
    
    def rsim(self, tsteps, ncells, i, sim_args = None):
        '''
            Run an individual simulation (with no initialisation) for parameters i 
                and pre-set seed for set numbers of time-steps and number of cells
        '''
        
        if sim_args is None:
            sim_args = {}
        
        tiss    = self.model(ncells = ncells, **self.model_params(i))
        results = tiss.simFull(tsteps = tsteps, **sim_args, **self.sim_params(i))
        results['tiss'] = tiss  
        
        return results
    
    

    def msim(self, tsteps, ncells, i, relaxsteps = 0, homeosteps = 0, sim_args = None,
             mut_args = None):
        ''' 
            Implement a mutated simulation

            Default is a single mutant cell
        '''
        if sim_args is None:
            sim_args = {}

        if mut_args is None:
            print('No mutant arguments specified, using middle cell')
            mut_args = {'middle_cell': True}

        if 'from_popidx' not in mut_args:
            print('Mutant population index not specified, using 0')
            mut_args['from_popidx'] = 0


        tiss    = self.model(ncells = ncells, **self.model_params(i))

        if relaxsteps > 0:
            tiss.sim(tsteps = relaxsteps, **sim_args, **self.sim_params(i))

        if homeosteps > 0:
            results = tiss.simFull(tsteps = homeosteps, **sim_args, **self.sim_params(i))
            tiss.reset()

        tiss.addMutatedPopulation(**mut_args)
        assert(tiss.poppts[-1] > 0), 'Mutated population must have at least one cell'


        results = tiss.simFull(tsteps = tsteps, **sim_args, **self.sim_params(i))
        results['tiss'] = tiss

        return results     
    
    
    def hsim(self, tsteps, ncells, i, relaxsteps = 0, homeosteps = 1000, 
             repeat_relax = False, sim_args = None):
        '''
            Runs an individual homeostatic simulation
                First run without cell decisions for relaxsteps
                Then run for homeosteps with cell decisions
                If repeat_relax:
                    check at homeostasis and repeat relaxation process until homeostasis is reached
                
            Assumed that the seed is already set
        '''
        if sim_args is None:
            sim_args = {}
        
        tiss    = self.model(ncells = ncells, **self.model_params(i))
        if relaxsteps > 0:
            tiss.sim(tsteps = relaxsteps, **sim_args, **self.sim_params(i))
            
        results = tiss.simFull(tsteps = homeosteps, **sim_args, **self.sim_params(i))
        if repeat_relax:
            while self.test_homeostasis(results['ncells']) == False:
                tiss.reset()
                results = tiss.simFull(tsteps = homeosteps, **sim_args, **self.sim_params(i))
        tiss.reset()
        results = tiss.simFull(tsteps = tsteps, **sim_args, **self.sim_params(i))
        results['tiss'] = tiss
        return results

       
    def asim(self, tsteps, ncells, i, relaxsteps = 1000, homeosteps = 1000, ctrsteps = None, repeat_relax = False, 
             kill_fn = None, kill_params = None, sim_args = None, deathscale = 1,
             tscale = 1, dshift = None):
        '''
            Runs an individual ablation simulation
                First run without cell decisions for relaxsteps
                Then run for homeosteps with cell decisions (runs twice), 
                If repeat_relax:
                    check at homeostasis and repeat relaxation process until homeostasis is reached
                
            Assumed that the seed is already set
            
            To apply a profile of cell death we need to apply a kill_fn:
                Of form kill_fn(t, tscale, dshift, tmin, otherparams, ...)
                where tmin is the minimimum time where the death-profile is applied in days
                Then killparams contains values of {dshift, tmin, otherparams} sets which values
                    you do not want to take their default values
            
            To use asim then tscale must be a specified parameter, and we can no longer
                re-scale time as a post-processing step.
        '''
        
        if kill_params is None:
            kill_params = {}
            
        if sim_args is None:
            sim_args = {}
        
        # Set-up control
        if ctrsteps is None:
            ctrsteps = homeosteps 
        control = {}
        
        #Get tscale and tshift
        if 'tscale' in self.allParamVars:
            tscale = self.paramset(i)['tscale']
        else:
            print('Warning: tscale not set. Death-profile may not match physical time.')
            
            
        #Handle inclusion of negative tscalevals
        if tscale < 0:
            assert('tau' in self.paramset(i))
            tscale = -self.paramset(i)['tau']/tscale
            #print('Assumes zero refractory period') #It doesn't if sim_model is set-up propery but be careful
            
        #dshift takes default value if set.
        #Note this is different to the regular tshift (offset by 1)
        if 'dshift' in self.allParamVars:
            kill_params['dshift'] = self.paramset(i)['dshift']
        else:
            if 'dshift' not in kill_params:
                if dshift is None:
                    if hasattr(self, 'tshift'):
                        print('Warning: using tshift as dshift')
                        kill_params['dshift'] = self.tshift
                    else:
                        kill_params['dshift'] = 0
                            
        
        #Set-up kill profile
        if 'deathscale' in self.allParamVars:
            deathscale = self.paramset(i)['deathscale']
            if deathscale < 0:
                deathscale = -deathscale/tscale
        else:
            print('Warning no death-scale provided, using 1')
            
        if kill_fn is not None:
            killprofile = deathscale*kill_fn(np.arange(tsteps), tscale = tscale, **kill_params)
        else:
            killprofile = None
        
        #Set-up model and run initialisation simulations
        tiss    = self.model(ncells = ncells, **self.model_params(i))
        tiss.sim(tsteps = relaxsteps, **sim_args, **self.sim_params(i))
        results = tiss.simFull(tsteps = homeosteps, **sim_args, **self.sim_params(i))
        if repeat_relax:
            while self.test_homeostasis(results['ncells']) == False:
                tiss.reset()
                results = tiss.simFull(tsteps = homeosteps, **sim_args, **self.sim_params(i))
        
        #Run control simulations and save relevant quantities
        tiss.reset()
        results = tiss.simFull(tsteps = ctrsteps, **sim_args)
        control['density'] = np.mean(np.sum(results['ncells'],1))
        control['divrate'] = np.mean(ds.simDivrate(tiss, results)[50:-50])
        
        #Run ablation simulation
        tiss.reset()
        results = tiss.simFull(tsteps = tsteps, **sim_args, killprofile = killprofile)
        results['tiss'] = tiss
        results['control'] = control
        
        return results          
        
    
       
    def update_maxseed(self, tsteps, ncells, seeds = 0, savefolder = '.', i = None, simtype = None):
        '''
            Calculates the maximum seed given fullparams and savefolder for each file
            
            seed: -1 (seeds to i), >= 0 (seeds to number), 
                if list or array runs array of seeds
                if None, then does not set seed, e.g. if you set seed in params
                
            simtype: 'h'/'m'
            
            i: i = some parameter id then we determine max seed for that
                otherwise we loop
        '''
        
        self.set_simtype(simtype)
            
        if seeds is None:
            seedmanage = False
        else:
            seedmanage = True
            if np.ndim(seeds) == 0:
                seeds = [seeds]
            else:
                seeds = seeds   
            
        if not os.path.exists(savefolder):
            return

        #Set-up runlist
        if i >= 0:
            runlist = [i]
        else:
            runlist = range(self.npsets)            
            
        for i in runlist:            
            for s in seeds:
                sn = self.sim_save_name(tsteps, ncells, i, s) 
                if not os.path.exists(savefolder + '/' + sn):
                    break

                #Update maximum seed
                if seedmanage and s > self.searchspace.iloc[i]['maxseed']:
                    self.update_param_seed(i, s)
       
       
    def to_skip_run(self, i, seeds):
        '''
            Determines whether to skip run, 
                Defaults to False 
        '''
        return False
    
    
    def run_sim(self, tsteps, ncells, i = -1, seeds = 0, savefolder = '.', 
                mergefirst = None, merge_during = False, save_every = None, 
                sim_args = None, init_params = None, parallel = None, ncores = 1,
                par_seed = True, end_save = True, norun = False,
                skip = 10, run_only = False, save_only_run = True, **kwargs):
        '''
            ncells is the initial number of cells
            To specify unequal # of initial cells per type input ncells as a list
            
            Key word arguments: include p0 (defaults to 3.5)
            
            seed: -1 (seeds to i), >= 0 (merge_searchspaceseeds to number), 
                if list or array runs array of seeds
                if None, then does not set seed, e.g. if you set seed in params
                
            merge_start: indicates whether we merge with saved at the start 
                of running
            save_every indicates we should save the searchspace as we go,
                save_every = None does not save as we go.
            merge_during: indicates whether when saving as we go or at the end
                we merge with the save-space
                In such a situation, existing run parameters are prioritised (1st in order)
                    followed by parameters already run, but not run on this search-space (3rd in order)
                    followed by unrun portion of search-space (2nd in order)
                    
            norun = False, only loads but does not run
            
            save_only_run = True: only saves the run seeds, and does so to a seeded
                searchspace
                    
            Run in parallel
            parallel: use the joblib.parallel object if you want to run in parallel
                else set None
            ncores: specifies to run simulation in parallel
            par_seed: True: parallises along seed axis, False: parallelises along i axis
            We just specify ncores and the function looks after itself
            
            Example: kwargs
                tscalevals

        '''
        
        if sim_args is None:
            sim_args = {}
            
        if init_params is None:
            init_params = {}
        
        #Flag for whether or not to process simulations
        self.run_only = run_only
        
        #Run in parallel
        if ncores > 1 and parallel is None:
            with Parallel(n_jobs = ncores, prefer="threads") as parallel:
                if par_seed:
                    print('Running with seeds in parallel with ncores =', ncores)
                    self.run_sim(tsteps, ncells, i, seeds, savefolder, mergefirst, merge_during, 
                        save_every, sim_args, init_params, parallel = parallel, norun = norun,
                        skip = skip, save_only_run = save_only_run, **kwargs)
                else:
                    print('Running with sims in parallel with ncores =', ncores)                    
                    parallel(delayed(self.run_sim)(tsteps, ncells, i, seeds, savefolder, mergefirst,  
                        merge_during = False, save_every = -1, sim_args = sim_args, init_params = init_params, 
                        parallel = parallel, par_seed = False, end_save = False,
                        norun = norun, skip = skip, save_only_run = save_only_run, **kwargs) for i in range(self.npsets))
                    
                return #We run with different settings
        
        #Set-up save-folder
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
            
        #Set-up merging parameters            
        if mergefirst is None:
            mergefirst = self.mergefirst
            
        if mergefirst:
            self.merge_searchspace(tsteps, ncells, savefolder, old_first = True)
            self.mergefirst = False
            
        if save_every is None:
            merge_during = False
        
        #Set-up runlist
        if i >= 0:
            runlist = [i]
        else:
            runlist = range(self.npsets)
            save_only_run = False
            
        #Set-up seeds
        if seeds is None:
            seedmanage = False
        else:
            seedmanage = True
            if np.ndim(seeds) == 0:
                seeds = [seeds]
            else:
                seeds = seeds
                    
            
        #Running iterations
        for i in runlist:
            print('Running for:', self.paramset(i))
            
            if self.to_skip_run(i, seeds):
                print('Skipping:', self.paramset(i))
                continue
            
            sim_data_by_seed = self.init_process_sims(seeds, i, tsteps, skip = skip, **kwargs)
            if parallel is None or not par_seed:
                for s in seeds:
                    sim_data_by_seed = self.run_sim_by_seed(i, s, sim_data_by_seed, tsteps, ncells, savefolder, 
                                        seedmanage, sim_args, init_params, norun = norun, skip = skip)
            else:
                outputs = parallel(delayed(self.run_sim_by_seed)(i, s, sim_data_by_seed.subdataset(s), tsteps, ncells, savefolder, 
                                        seedmanage, sim_args, init_params, norun = norun, skip = skip) for s in range(len(seeds)))
                self.update_maxseed(tsteps, ncells, seeds, savefolder, i) #update max seed
                #Merges the seeded datasets into the non-seeded ones
                if self.run_only == False:
                    sim_data_by_seed = ds.join_datasets([sim_data_by_seed] + outputs)
            self.process_sims(sim_data_by_seed, i, **kwargs)
            
            if merge_during and (i + 1) % save_every == 0:
                self.merge_run_portion(tsteps, ncells, i, savefolder, save = True)
            
        
        if end_save:
            if save_only_run:
                self.use_indexed_ss_prefix(runlist[0])
                self.save_searchspace(tsteps, ncells, savefolder + '/searchspaces/', merge = False, sub_indices = np.array(runlist))
                #self.merge_indexed_searchspace(np.arange(len(self.searchspace)), tsteps, ncells, savefolder)
            else:
                self.save_searchspace(tsteps, ncells, savefolder, merge = True)
            
            
        
        
    def get_sim_data(self, p, tsteps, ncells, savefolder = '.', seeds = None, nseeds = 1, 
                     alltimes = True, norm_by_control = [], skip = 10, shed_profile = None,
                     sbasrefract = 0.0, shedvals = [0], norun = True,
                     sbas_cv = 1.0, init_params = None, **kwargs):
        '''
            Get simulation data by parameter set p
            
            Use tsteps, ncells, seeds as defined else-where
            
            Assumes the simulation is already run
            
            To extract by tscaleval and shedval
            
            We can re-run with an arbitrary shed_profile by specifying
                shed_profile is not None
                
            
        '''
        
        if seeds is None:
            seeds = np.arange(nseeds)
        
        sim_data_by_seed = self.init_process_sims(seeds, p, tsteps, alltimes = alltimes, 
                                                  norm_by_control = norm_by_control, skip = skip, 
                                                  use_exp_bins = False, sbasrefract = sbasrefract,
                                                  shedvals = shedvals, **kwargs)
        
        for s in seeds:
            sim_data_by_seed = self.run_sim_by_seed(p, s, sim_data_by_seed, tsteps, ncells, savefolder, 
                                        seedmanage = False, verbose = False, norun = norun, skip = skip,
                                        shed_profile = shed_profile, init_params = init_params)
            if shedvals[0] != 0:
                print('shed:', shedvals[0], 'refract:', sbasrefract, 'cv:', sbas_cv)
                sim_data_by_seed.rerun_suprabasal(shedvals[0], sbasrefract, seed = s, sbas_cv = sbas_cv)
            
            
        return sim_data_by_seed
        
        
    def get_best_fit(self, metric, tsteps, ncells, savefolder = '.', nseeds = 1, save = False,
                     constraints = None, norm_by_control = [], shed = None, print_fit = True, 
                     return_seeded = False, post_opt = None, norun = True, 
                     init_params = None, **kwargs):
        '''
            Gets sim-data object for the best fit to a file,
                if save = True then we save to a file
                
            We can specify constraints, by using a boolean array of length of search-sapce
        '''

        if post_opt is None:
            post_opt = {}
        
        params = self.best_fit(metric, constraints)
        
        if print_fit:
            print('Best fit:', params)
        else:
            print('tscale:', params.tscale)

        if params['index'] != params.name:
            #probably is fine as we don't use index anywhere except for seeding
            print('Warning: index is not the same as name')
        p = int(params.name)
        
        if shed is not None:
            params.shed = shed
            
        #set post-optimisation params
        if (len(self.postopt_params.columns) > 0) and (metric + ':' + self.postopt_params.columns[0]) in params:
            poptparams = pd.Series()
            for pi in self.postopt_params.columns:
                if (metric + ':' + pi) in params:
                    poptparams[pi] = params[metric + ':' + pi]
            params[metric + ':postopt'] = poptparams
            post_opt = {**poptparams, **post_opt}
        elif metric + ':postopt' in params:
            if np.ndim(params[metric + ':postopt']) == 0 and np.isnan(params[metric + ':postopt']):
                params[metric + ':postopt'] = pd.Series()
            post_opt = {**params[metric + ':postopt'], **post_opt} #overwrite with post-opt            

                          
        sim_data_by_seed = self.get_sim_data(p, tsteps, ncells, savefolder, nseeds = nseeds,
                                     tscalevals = [params.tscale], shedvals = [params.shed],
                                     norm_by_control = norm_by_control, norun = norun, 
                                     init_params = init_params, **post_opt, **kwargs)
        
        if return_seeded:
            sim_data = sim_data_by_seed
        else:
            sim_data = sim_data_by_seed.calculate_combined_quantities(full = True)
        
        
        if save:
            bfitname = 'best_' + metric + '_' + self.searchspace_save_name(tsteps, ncells)
            ds.save_object(sim_data, savefolder + '/' + bfitname)
        
        sim_data.run_params = params

        return sim_data
        
        
        
    def merge_run_portion(self, tsteps, ncells, i, savefolder = '.', ss_prefix = None, save = True):
        '''
            Merges save-space with existing run portion
            
            We merge the search-space in a sequential way:
                We change search-space order, save the run portion on this simulator
                Then we save the un-run portion of this simulator (not priotised)
                Then we save the loaded already run portion (prioritised 2ndd)
        '''
                
        curr_searchspace = self.searchspace
        
        #Merge with old if it exists
        if os.path.exists(savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix)):
            saved_space = self.load_searchspace(tsteps, ncells, savefolder, ss_prefix)
            
            #Merge unrun with saved, keeping saved
            curr_searchspace = pd.concat([self.searchspace[(i+1):], saved_space])
            curr_searchspace = curr_searchspace.drop_duplicates(subset = self.allParamVars, keep = 'last')
            curr_searchspace = pd.concat([self.searchspace[:(i+1)], curr_searchspace])
            curr_searchspace = curr_searchspace.drop_duplicates(subset = self.allParamVars, keep = 'first')
            curr_searchspace = curr_searchspace.reset_index(drop = True) # drop = True

        self.searchspace = curr_searchspace
        
        if save:
            self.save_searchspace(tsteps, ncells, savefolder)
           
        return  self.searchspace  
        
        
        

    def on_runlist(self, paramset, remove_duplicates = False):
        '''
            Check if the parameter set is already on the run-list
            
            Based on: https://stackoverflow.com/questions/24761133/pandas-check-if-row-exists-with-certain-values
        '''
        
        fullpset = {**self.fixedParams, **paramset} #paramset prioritised
        
        dp  = pd.DataFrame([list(fullpset.values())], columns = list(fullpset.keys()))
        if remove_duplicates:
            self.remove_parameter_duplicates()
         
        return bool(pd.concat([self.fullparams,dp]).shape[0] - pd.concat([self.fullparams,dp]).drop_duplicates().shape[0])


    def getParamRow(self, paramset, remove_duplicates = False):
        '''
            Extract the row a given parameter set appears in the dataframe fullparams
        '''
        
        if remove_duplicates:
            self.remove_parameter_duplicates()
        
        fullpset = {**self.fixedParams, **paramset} #paramset prioritised        
                
        row = np.array(self.fullparams.query(dict_to_dfquery(fullpset)).index)
        if len(row) == 0:
            return None
        elif len(row) == 1:
            return row[0]
        else:
            print('Duplicates found')
            return row[0]
     
    """
    def getOverlappingRows(self, df1):
        '''
            Gets overlapping rows between fullparams and another dataframe, e.g. search space
        '''
        return
    """
        
    def load_searchspace(self, tsteps, ncells, savefolder = '.', ss_prefix = None):
        '''
            Search space contains a dataframe with the range of parameters explored that we have saved
            
            We load the existing run space
            
        '''       
        return ds.load_object(savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix))


    def merge_searchspace(self, tsteps, ncells, savefolder = '.', ss_prefix = None, old_first = False,
                          remove_duplicates = True):
        '''
            Search space contains a dataframe with the range of parameters explored that we have saved
            
            We merge the existing simulations run
        '''
        
        new_searchspace = self.searchspace
        
        #Merge with old if it exists
        if os.path.exists(savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix)):
            current_searchspace = self.load_searchspace(tsteps, ncells, savefolder, ss_prefix)
            if old_first:
                new_searchspace = pd.concat([current_searchspace, self.searchspace])
            else:
                new_searchspace = pd.concat([self.searchspace, current_searchspace])
            #new_searchspace = new_searchspace.drop_duplicates(subset = self.allParamVars)

        self.searchspace = new_searchspace
        
        if remove_duplicates:
            self.remove_parameter_duplicates()
        self.searchspace = self.searchspace.reset_index(drop = True)
           
        return  new_searchspace  
    

    def save_searchspace(self, tsteps, ncells, savefolder = '.', merge = False, ss_prefix = None, 
                         sub_indices = None):
        '''
            Search space contains a dataframe with the range of parameters explored that we have saved
            
            We merge the existing simulations run
            
            sub_indices: specifies if we want to save a subset of the searchspace
        '''   
        
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
             
        if merge:
            self.merge_searchspace(tsteps, ncells, savefolder, ss_prefix = ss_prefix, old_first = False)

        if sub_indices is not None:
            searchspace = self.searchspace.iloc[sub_indices]
        else:
            searchspace = self.searchspace

        ds.save_object(searchspace, savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix))
        print('Saving search space to:', self.searchspace_save_name(tsteps, ncells, ss_prefix))
        return  self.searchspace  
    
    
    def repair_searchspace(self, tsteps, ncells, savefolder = '.'):
        '''
            Search space contains a dataframe with the range of parameters explored that we have saved        
        
            Checks simulations have not been deleted, if they have we remove from dataframe
            
            If we want to re-run missing simulations then use the relevant function to run them.
        '''
        
        current_searchspace = self.load_searchspace(tsteps, ncells)
        current_searchspace = current_searchspace.reset_index(drop = True)
        
        changed = False
        for p in range(len(current_searchspace.index)):
            sn = self.sim_save_name(tsteps, ncells, p, seed = 0, paramsdf = current_searchspace)
            if not os.path.exists(savefolder + '/' + sn):
                print('Missing sim', p, current_searchspace.iloc[p])
                current_searchspace.drop(p)
                changed = True                    
        
        current_searchspace = current_searchspace.reset_index(drop = True)        
        
        if changed == True:
            ds.save_object(current_searchspace, self.searchspace_save_name(tsteps, ncells))
            
        return  current_searchspace          
        
    #def clear_searchspace(self, tsteps, ncells, savefolder = '.'):
        
        
    def merge_indexed_searchspace(self, indices, tsteps, ncells, savefolder = '.', save = True,
                                 ss_prefix = None, old_first = True, use_subfolder = True):
        '''
            Merges search-space following indexing convention with other 
                parallel run indices
        
        '''
        
        self.merge_seeded_searchspace(indices, tsteps, ncells, savefolder, save, ss_prefix, 
                                      old_first, use_index = True, use_subfolder = use_subfolder)



    def merge_seeded_searchspace(self, move_seeds, tsteps, ncells, savefolder = '.', save = True,
                                 ss_prefix = None, old_first = False, use_index = False, use_subfolder = False):
        '''
            Merges search-space following seeding convention with other 
                parallel run seeds
        
        '''
        
        if np.ndim(move_seeds) == 0:
            move_seeds = [move_seeds]
            
        self.use_unseeded_ss_prefix()
        self.merge_searchspace(tsteps, ncells, savefolder, ss_prefix, old_first, remove_duplicates = False)
            
        if use_subfolder:
            sf = savefolder + '/searchspaces/'
        else:
            sf = savefolder
            
        for s in np.sort(np.array(move_seeds))[::-1]:
            if use_index:
                self.use_indexed_ss_prefix(s)
            else:
                self.use_seeded_ss_prefix(s)
                
            self.merge_searchspace(tsteps, ncells, sf, ss_prefix, 
                                   old_first, remove_duplicates = False)
        
        self.remove_parameter_duplicates()
        self.use_unseeded_ss_prefix()
        
        if save:
            self.setup_sweeps() # this deletes all sweeps
            self.save_searchspace(tsteps, ncells, savefolder)
        
    
    

class optimizer(simulator):
    '''
        optimizer class:
            The default behaviour is a brute force optimizer by running simulations over multiple sets of paramemters and 
            comparing with data
        
            The sub-classes: sim_anneal, gd, etc. can run simulated annealing, gradient descent, etc.
            
    '''    
    
    def __init__(self, model, freeParams, exp_file, params = None, metrics = None, #metrics
                 simtype = '', mergefirst = True, move_seed = None, exp_prefix = '', 
                 exp_suffix = '', tshift = 0.0, use_int = [], dp_round = 8, toCalculate = None,
                 order_params = None, compute_kwargs = None, max_time = np.inf):
        
        #Set-up metrics
        self.setup_metric(metrics) 
        
        #Sets up search space and model
        simulator.__init__(self, model, freeParams, params, move_seed, simtype, mergefirst, 
                           use_int = use_int, dp_round = dp_round, order_params = order_params, compute_kwargs = compute_kwargs)        
        
        #Update to calculate based on metrics
        self.toCalculate = self.getToCalculate(self.metrics, toCalculate, order_params)
        
        self.load_exp_data(exp_file, exp_prefix, exp_suffix, tshift, compute_kwargs, max_time)

        if exp_prefix != '':
            exp_prefix = exp_prefix + '_'

        self.searchspace_prefix = 'fits_' + exp_prefix + self.exp_data.filename
                
        #Use saved metrics
        self.use_saved_metrics = False 
        
        #Set-up tscale-vals and shedvals, defaults to None
        self.tscalevals = [1]
        self.shedvals   = [0]
        self.postopt_params = pd.DataFrame() #Dataframe storing post-optimisation parameters, we loop over each row
        
        self.tsc_opt    = 'grid'
        self.shed_opt   = 'grid'
        self.nboots = 0
        
        self.run_only = False #flag to not process simulations
        
        
    def load_exp_data(self, exp_file, exp_prefix, exp_suffix, tshift, compute_kwargs = None, max_time = np.inf):
        #Load experimental data
        self.exp_data = ds.exp_dataset(exp_file, exp_prefix, exp_suffix, self.toCalculate, compute_kwargs = compute_kwargs,
                                       max_time = max_time)
        self.exp_data.reshape_pdfs(2) #re-shapes pdfs to have 2 zeros as last elements
        self.tshift = tshift #time-shift in experimental units between simulation & experiment               
        
        
    @property
    def runspace(self):
        cols = self.allParamVars + ['maxseed']
        if 'index' in self.searchspace:
            cols = ['index'] + cols
        return self.searchspace[cols]
    
    
    def use_unseeded_ss_prefix(self):
        '''
            Revert to standard prefixing method
        '''
        ss_exp_prefix = self.exp_data.exp_prefix
        if ss_exp_prefix != '':
            ss_exp_prefix = ss_exp_prefix + '_'                
        self.searchspace_prefix = 'fits_' + ss_exp_prefix + self.exp_data.filename 


    def use_seeded_ss_prefix(self, s = None):
        '''
            Naming convention for multiple parallel runs of fitting algorithm
        '''
        if s is None:
            s = self.move_seed 
        
        ss_exp_prefix = self.exp_data.exp_prefix
        if ss_exp_prefix != '':
            ss_exp_prefix = ss_exp_prefix + '_'                
        self.searchspace_prefix = 'fits_' + ss_exp_prefix + self.exp_data.filename + '_seed_' + str(s)


    def use_indexed_ss_prefix(self, i = None):
        '''
            Naming convention for multiple parallel runs of fitting algorithm
        '''
        if i is None:
            i = self.move_seed 
        
        ss_exp_prefix = self.exp_data.exp_prefix
        if ss_exp_prefix != '':
            ss_exp_prefix = ss_exp_prefix + '_'                
        self.searchspace_prefix = 'fits_' + ss_exp_prefix + self.exp_data.filename + '_index_' + str(i)

    
    
        
    def to_skip_run(self, i, seeds):
        '''
            Determines whether or not to skip the process of fitting a certain 
                parameter set. Ensures that the right number of seeds are run
                and all metrics in metrics have non-default, nan or inf values
        '''
        
        if not self.use_saved_metrics:
            return False
        
        self.searchspace.at[i, 'maxseed'] = len(seeds)
        
        skiprun = True
        for m in self.metrics:
            if (self.searchspace.at[i, m.name] == self.ss_defaults[m.name]) or np.isnan(self.searchspace.at[i, m.name]):
                skiprun = False
                break
        
        return skiprun
        
    
    def format_tscalevals(self):
        if self.tscalevals is None: 
            self.tscalevals = [1]
        elif np.ndim(self.tscalevals) == 0:
            self.tscalevals = [self.tscalevals]
            
        self.tscalevals = np.sort(np.array(self.tscalevals))        
        if self.tscalevals is None: 
            self.tscalevals = [1]
        elif np.ndim(self.tscalevals) == 0:
            self.tscalevals = [self.tscalevals]
        
        self.tscalevals = np.round(self.tscalevals, self.dp_round)
        return self.tscalevals

        
    def format_shedvals(self):
        if self.shedvals is None:
            self.shedvals = [0]  
        elif np.ndim(self.shedvals) == 0:
            self.shedvals = [self.shedvals]
        
        if np.ndim(self.shedvals[0]) == 0:
            self.shedvals   = np.sort(np.array(self.shedvals))
            
        self.shedvals = np.round(self.shedvals, self.dp_round)
        return self.shedvals
                
                
    
    def setup_postopt(self, tscalevals = None, shedvals = None, tsc_opt = None, shed_opt = None, save_sweeps = False,
                      nboots = 0, boot_exp = True, boot_seed = 0, tsteps = None, ncells = None, savefolder = '.', 
                      seeds = None, sim_boot_metric = None, postopt_params = None, **kwargs):
        '''
            Sets up post-run optimiser
        '''     

        self.tscalevals = tscalevals
        self.shedvals   = shedvals

        if postopt_params is not None:
            self.setup_postopt_params(postopt_params)
        
        self.tsc_opt    = tsc_opt
        self.shed_opt   = shed_opt
        
        if not self.needsSuprabasal():
            self.shed_opt = 'grid' #we have no suprabasal optimisation needed
            self.shedvals = None   
            
        self.format_tscalevals()
        self.format_shedvals()
        
        #If we save save_sweeps
        self.save_sweeps = save_sweeps
        if save_sweeps:
            self.setup_sweeps()
            #self.use_saved_metrics = False
            #print('Saving sweeps, use_saved_metrics set to False')
        
        # Set-up bootstrapping
        if nboots > 0:
            if not boot_exp:
                print('Bootstrapping best simulated dataset')
                assert(sim_boot_metric in self.metric_names)
                sim_data = self.get_best_fit(sim_boot_metric, tsteps, ncells, savefolder = savefolder, nseeds = len(seeds))             
                self.setup_bootstrap(sim_data, nboots, boot_seed)
            else:
                self.setup_bootstrap(None, nboots, boot_seed)
  
  
  
    def setup_sweeps(self):
        '''
            Sets up sweeps for optimisation
        '''
        
        for m in range(len(self.metrics)):
            metname = self.metrics[m].name
            self.searchspace[metname + ':sweeps']  = None          


        
    def setup_bootstrap(self, dataset = None, nboots = 100, boot_seed = 0, maxcost = 1e10): 
        '''
            Sets up bootstrap optimiser
        '''
        
        if dataset is None:
            dataset = self.exp_data
            print('Bootstrapping experiment data')
            
        self.boot_data = dataset.bootstrap(nboots, boot_seed)
        self.nboots = nboots
        
        
        for m in range(len(self.metrics)):
            metname = self.metrics[m].name
            self.searchspace[metname + ':boot_costs']  = None
            self.searchspace[metname + ':boot_tscale'] = None
            self.searchspace[metname + ':boot_shed']   = None          
                    
        
        #Now captured in search-space
        #self.boot_costs   = np.ones((self.npsets,  self.nboots, len(self.metrics)))*maxcost
        #self.boot_tscales = np.ones((self.npsets,  self.nboots, len(self.metrics)))
        #elf.boot_sheds   = np.zeros((self.npsets, self.nboots, len(self.metrics)))
        
        #self.use_saved_metrics = False
        
        
        
    def optimise(self, tsteps, ncells, seeds = None,
                 savefolder = '.',
                 shedvals   = None, shed_opt = 'grid',
                 tscalevals = None, tsc_opt  = 'grid', mergefirst = None, 
                 use_saved_metrics = False, sim_args = None, init_params = None,
                 save_to_runspace = True, postopt_params = None, **kwargs):
        '''
            Default optimiser is grid_optimise,
                for inputs see grid_optimise
        '''
                
        if sim_args is None:
            sim_args = {}
            
        if init_params is None:
            init_params = {}
        
        self.grid_optimise(tsteps, ncells, seeds,
                 savefolder, shedvals = shedvals, shed_opt = shed_opt, postopt_params = postopt_params,
                 tscalevals = tscalevals, tsc_opt  = tsc_opt, mergefirst = mergefirst, 
                 use_saved_metrics = use_saved_metrics, sim_args = sim_args,
                 init_params = init_params, **kwargs)
        
        #Saves to runspace
        if save_to_runspace:
            self.save_searchspace_to_runspace(tsteps, ncells, savefolder)
          
        
    def load_existing(self, tsteps, ncells, savefolder, keep_current = True, 
                      tscalevals = None, shedvals = None, postopt_params = None):
                      #save_sweeps = False, load_boots = False, boot_exp = True):
        '''
            Loads an existing search-space
            
            Has the ability to load sweep files and bootstrap files if they exist
        
        '''
        
        if not keep_current:
            self.remove_searchspace_rows()
        
        self.merge_searchspace(tsteps, ncells, savefolder, old_first = True) # get pre-fitted
        
        if tscalevals is not None:
            self.tscalevals = tscalevals
            self.format_tscalevals()
            
        if shedvals is not None:
            self.shedvals = shedvals
            self.format_shedvals()   
            
        if postopt_params is not None:
            self.setup_postopt_params(postopt_params)
        
        #Deprecated as now captured in searchspace
        '''
        if save_sweeps:
            self.load_sweeps(tsteps, ncells, savefolder)
            
        if load_boots:
            self.load_boots(tsteps, ncells, savefolder, boot_exp = boot_exp)
        '''


    def remove_searchspace_rows(self):
        self.searchspace = self.searchspace.iloc[0:0]
        print('Warning: removing rows of existing searchspace')
        
        
    def grid_optimise(self, tsteps, ncells, seeds = None,
                 savefolder = '.',
                 shedvals   = None, shed_opt = 'grid',
                 tscalevals = None, tsc_opt  = 'grid', 
                 postopt_params = None, mergefirst = None, 
                 use_saved_metrics = False, sim_args = None, init_params = None,
                 ncores = 1, par_seed = True, save_sweeps = False,
                 nboots = 0, boot_exp = True, boot_seed = 0, **kwargs):
        '''
            For a specified system size and simulation length we use optimisation object to optimise, 
                if exisiting simulations exist we use them, otherwise we run new ones
                
            tsteps: specify the number of simulations time-steps to use
            ncells: specify the initial number of cells to use for the simulation
            mode:   'r': normal run, 'h': homeostatic simulation, 'm': homeostatic then mutant simulation
                this is just simtype above
            seeds: seeds to run simulations for, otherwise will detect the number of seeds based on 
                the pre-run files

            
            For re-running of suprabasal layer we can use shedvals: 
                for None/0 we do not re-run suprabasal layer
                for negative, we multiply by the governing time-scale of the problem
                e.g. starts with div rate highest level population (but if doesn't exist we move down
                    population values)
                for positive, we run for these scale values in sim units
            shed_opt = 'grid'/'GN' specifies whether to use a discrete grid or Gauss-Newton method
            Alternatively: use shed_opt = 'grid'/'GN' to use a 
            
            Similar behaviour for tscale with tscalevals and tsc_opt 
            
            Parallelisation:
            To run simulations in parallel use ncores > 1, this only allows parallisation
                alongside the seed axis        
            If you set par_seed = True
            
            
            Saving data for uncertainty analysis:
            save_sweeps: True/False, if True then we save the sweeps of tscale, shed
            
            To implement a bootstrap analysis: use nboots > 0
                nboots: number of bootstrap samples
                boot_exp: True/False, if True then we bootstrap the experimental data
                    otherwise we bootstrap the simulated data
                If we bootstrap from simulated dataset, we need to provide sim_boot_metric
                
            
        '''

        if sim_args is None:
            sim_args = {}
            
        if init_params is None:
            init_params = {}    

        #Set-up saving and merging of optimiser
        self.setup_optimiser(tsteps, ncells, savefolder, mergefirst, use_saved_metrics)
        
        #Set-up post-prpocessing optimisers
        self.setup_postopt(tscalevals, shedvals, tsc_opt, shed_opt, save_sweeps,
                           nboots, boot_exp, boot_seed, tsteps, ncells, savefolder, seeds,
                           postopt_params = postopt_params, **kwargs)
        
        
        # Run simulations
        self.run_sim(tsteps, ncells, seeds = seeds, savefolder = savefolder, sim_args = sim_args, init_params = init_params,
                        ncores = ncores, par_seed = par_seed, save_sweeps = save_sweeps, **kwargs)
        
        
        '''
        Deprecated: now captured in searchspace
        
        if save_sweeps:
            self.save_sweeps_to_file(tsteps, ncells, savefolder)
            
        if nboots > 0:
            self.save_boots(tsteps, ncells, savefolder)
        '''
            
        
    def save_sweeps_to_file(self, tsteps, ncells, savefolder = '.'):
        '''
            Saves the sweeps of tscale, shed
        '''
        sweeps = {'sweeps': self.sweeps, 'tscalevals': self.tscalevals, 'shedvals': self.shedvals,
                  'postopt_params': self.postopt_params}

        ds.save_object(sweeps, savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix = 'fullsweeps_' + self.exp_data.exp_prefix + '_' + self.exp_data.filename))
        print('Saving sweeps to:', self.searchspace_save_name(tsteps, ncells, ss_prefix = 'fullsweeps_' + self.exp_data.exp_prefix + '_' + self.exp_data.filename))
    

    def save_boots(self, tsteps, ncells, savefolder = '.'):
        '''
            Saves bootstrapping metrics of simulations to file
        '''
        
        boots = {'boot_costs': self.boot_costs, 'boot_tscales': self.boot_tscales, 'boot_sheds': self.boot_sheds,
                 'nboots': self.nboots, 'bootmode': self.boot_data.type}
        
        ds.save_object(boots, savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix = 'boot' + self.boot_data.type))
        print('Saving boots to:', self.searchspace_save_name(tsteps, ncells, ss_prefix = 'boot' + self.boot_data.type))
        
        
    
    def load_sweeps(self, tsteps, ncells, savefolder = '.'):
        """
        Load the sweeps saved by save_sweeps_to_file, only if the file exists.
        
        Args:
            tsteps (int): Number of time steps.
            ncells (int): Number of cells.
            savefolder (str): Path to the folder where the sweeps are saved. Default is current directory.
        """
        
        filename = savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix= 'fullsweeps_' + self.exp_data.exp_prefix + '_' + self.exp_data.filename)
        
        if os.path.exists(filename):
            sweep_file = ds.load_object(filename)
            self.sweeps = sweep_file['sweeps']
            self.tscalevals = sweep_file['tscalevals']
            self.shedvals = sweep_file['shedvals']
            if 'postopt_params' in sweep_file:
                self.postopt_params = sweep_file['postopt_params']
            print("Sweeps loaded successfully.")
        else:
            print("Sweeps file does not exist.")
            
            
    def load_boots(self, tsteps, ncells, savefolder = '.', boot_exp = True):
        '''
            Loads the bootstrapped metrics saved by save_boots, only if the file exists.
        '''
        
        if boot_exp:
            bootmode = 'exp'
        else:
            bootmode = 'sim'
        
        filename = savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix = 'boot' + bootmode)
        
        if os.path.exists(filename):
            boots_file = ds.load_object(filename)
            self.boot_costs = boots_file['boot_costs']
            self.boot_tscales = boots_file['boot_tscales']
            self.boot_sheds = boots_file['boot_sheds']
            self.nboots = boots_file['nboots']
            print("Boots loaded successfully.")
            
            if bootmode == 'sim':
                print('Warning: bootstrapping from simulated data, bootstrapped means may be smaller due to truncation')
        else:
            print("Boots file does not exist.")
            
        
    def setup_optimiser(self, tsteps, ncells, savefolder = '.', mergefirst = None, use_saved_metrics = False):
        #Sets whether to re-calculate anything
        self.use_saved_metrics = use_saved_metrics

        #Set-up save-folder
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        #Merge existing
        if mergefirst is not None:
            self.mergefirst = mergefirst
            
        if self.mergefirst:
            self.merge_searchspace(tsteps, ncells, savefolder, old_first = True) # get pre-fitted
            self.merge_runspace_to_searchspace(tsteps, ncells, savefolder) #get pre-run
            self.mergefirst = False  
            #print(self.searchspace.to_string())      
        

    def merge_runspace_to_searchspace(self, tsteps, ncells, savefolder):
        '''
            Merge pre-run simulations into the search-space and fills gaps with nans
                Starts by merging search-space in case we have already fit certain 
                simulations
        '''
    
        #Updates from the saved run-space
        self.merge_searchspace(tsteps, ncells, savefolder, ss_prefix = 'runspace', old_first = False)
        self.ss_nans_to_defaults()
        
        #Updates from all in file, we lose the run-seed information
        self.update_runspace_from_files(tsteps, ncells, savefolder)
        self.merge_searchspace(tsteps, ncells, savefolder, ss_prefix = 'runspace', old_first = False)
        self.ss_nans_to_defaults()
        
    
    def update_runspace_from_files(self, tsteps, ncells, savefolder):
        ''' 
            Loads the runspace from which files have been saved
        '''
        
        matching_files = self.sim_save_glob(tsteps, ncells, seed = 0, savefolder = savefolder)
        new_pars = {p: [] for p in self.allParamVars}
        new_pars['index'] = []
        i = 0
        
        if len(matching_files) == 0:
            print('No matching files found')
            return
        
        for m in matching_files:
            m1 = m.split('/')[-1]
            
            for p in self.allParamVars:
                pval = m1.split('_' + p + '_')[1].split('_')[0]
                if is_number(pval):
                    pval = float(pval)
                elif is_bool(pval):
                    pval = bool(pval)
                    
                new_pars[p].append(pval)
            new_pars['index'].append(10000+i)
            i += 1
            
        runspace = pd.DataFrame.from_dict(new_pars)
        runspace['maxseed'] = -1
        
        ds.save_object(runspace, savefolder + '/' + self.searchspace_save_name(tsteps, ncells, 'runspace'))
        print('Saving updated runspace:', self.searchspace_save_name(tsteps, ncells, 'runspace'))
        
        return runspace
        

        
    def save_searchspace_to_runspace(self, tsteps, ncells, savefolder = '.', merge = False):
        '''
            Takes search-space and gets the run-space and saves to us
        '''
        ss_prefix = 'runspace'
        
        if merge:
            self.merge_searchspace(tsteps, ncells, savefolder, ss_prefix = ss_prefix)        
        
        runspace = self.runspace
        ds.save_object(runspace, savefolder + '/' + self.searchspace_save_name(tsteps, ncells, ss_prefix))
        print('Saving runspace:', self.searchspace_save_name(tsteps, ncells, ss_prefix))

        return runspace
    
       

        
        
    def set_ss_defaults(self, maxcost = 1e10):
        self.ss_defaults = {}        
        self.ss_defaults['maxseed'] = -1
        
        
        for m in self.metrics:
            self.ss_defaults[m.name] = maxcost    
            self.ss_defaults[m.name+':shed']    = 0.0
            self.ss_defaults[m.name+':tscale']  = 1.0   
            #self.ss_defaults[m.name+':postopt'] = pd.Series({})
            
             
        

    def setup_metric(self, metric = None, popnames = None):
        '''
            Specify a metric as the type of data averages
                metric = {cost = 'cost_prd' / 'cost_sum',
                        quantity = [] %choose from 'av_bas_cs' / 'av_tot_cs' / 'density' / 'divrate',
                        dist = [] % choose from 'bas_pdf' / 'sup_pdf' / 'tot_cs',
                        dist_metric = 'lsq' / 'kld' / 'ksd'
                        popOnly = -1 (use any population) / 0/1/2: specify population 
                                /[1,2]: specify multiple populations }
                        
            Specify as a list to calculate multiple metrics
        '''
        
        
        
        self.metrics = []
        
        if metric is None:
            metric = [{}]
        elif type(metric) is dict:
            metric = [metric]
        elif type(metric) is not list:
            print('Valid metric not provided')
            assert(0)
        
        #Create metric objects
        self.metrics = [metrics.metric(**m, popnames = popnames) for m in metric]
        
        return self.metrics
    
    
    @property
    def metric_names(self):
        mnames = []
        for m in self.metrics:
            mnames.append(m.name)
        return mnames
    
        
    def getToCalculate(self, metrics = None, toCalculate = None, order_params = None):
        '''
            Determine the quantities (including pdfs) to calculate from metrics
        '''
        
        if toCalculate is None:
            toCalculate = []
            
        if metrics is None:
            metrics = []
        
        toCalculate += cop.getToCalculate(order_params)
        
        toCalculate = set(toCalculate)
        
        for m in metrics:
            for q in m.allquantities():
                toCalculate.add(q)
        
        return list(toCalculate)
        
        
    def needsSuprabasal(self):
        '''
            Based on toCalculate determines whether we need to run the suprabasal layer
        '''
        #returns True if any common elements and false otherwise
        actual_cats = [c.split('.')[1] if '.' in c else c for c in self.toCalculate]
        return bool(set(ds.supra_cats) & set(actual_cats)) 
    
    
    def init_process_sims(self, seeds, i = None, tsteps = None, alltimes = True, 
                 norm_by_control = [], save_results = None,  skip = 10,
                 tscalevals = None, shedvals = None, use_exp_bins = True, 
                 sbasrefract = 0.0, **kwargs):
        
        '''
            Initialise simulation data by seed to create data saving in the right size
                for later processing
                
            i - is the index of the parameter set
        '''
        
        if self.run_only:
            return ds.dataset()
        
        
        if i is None:
            params = {}
        else:
            params = self.paramset(i)
                
                
        """
        #We no longer need this, as we no default to storing reduced data pre-computed
        #over tscale and shed
        
        # Set-up tscale grid search, if required
        if alltimes is None:
            if self.tsc_opt != 'grid':
                alltimes = True
            else:
                alltimes = False
            
        # Set-up shedding grid search, if required
        if save_results is None:
            if self.shed_opt != 'grid':
                save_results = True
            else:
                save_results = False 
        """
                
        #using defaults
        if tscalevals is None:
            tscalevals = self.tscalevals
        if shedvals is None:
            shedvals = self.shedvals
            
        if use_exp_bins:
            max_pdf_bin = self.exp_data.max_pdf_bin
        else:
            max_pdf_bin = None

        sim_data_by_seed = ds.sim_dataset(self.toCalculate, self.exp_data.times, tsteps, seeds, npops = self.npops(), 
                                          max_pdf_bin = max_pdf_bin, tscalevals = tscalevals, shedvals = shedvals,
                                          alltimes = alltimes, save_results = save_results, tshift = self.tshift,
                                          sim_params = params, norm_by_control = norm_by_control, skip = skip,
                                          sbasrefract = sbasrefract, analyse_extinct = self.analyse_extinct,
                                          compute_kwargs = self.compute_kwargs)
        #Re-shape pdfs according to experimental data, achieved by already passing the right data
        

        return sim_data_by_seed
        
    
    def process_sim_by_seed(self, sim_data_by_seed, results, seed = 0, **kwargs):
        '''
            This function allows processing of simulation, immediately after running
            In the future we can call an analysis script, or something. 
        '''
        
        if self.run_only:
            return sim_data_by_seed
        
        sim_data_by_seed = sim_data_by_seed.quantities_by_seed(results['tiss'], results, seed)

        return sim_data_by_seed
    
    
    def process_sims(self, sim_data_by_seed, i, shed_opt = 'grid', 
                     tsc_opt  = 'grid', postopt_settings = None, **kwargs):
        '''
            This function allows processing of simulations, after collecting 
                individual data by seed
                
            Option for optimisation on tscale and shed
        '''
        
        if self.run_only:
            return sim_data_by_seed        
        
        if sim_data_by_seed.extinct:
            self.set_to_extinct(i)
            if not self.analyse_extinct:
                return
        
        if postopt_settings is None:
            postopt_settings = {}
       
        
        if tsc_opt == 'grid':
            if shed_opt == 'grid':
                metric_data = self.calculate_grid_metrics(sim_data_by_seed, i, **postopt_settings)
                
            else: #grid on tscale and find best shed using GN
                metric_data = self.optim_tsc_grid_sb_gn()
                
        elif tsc_opt == 'GD' and shed_opt == 'GD':
            metric_data = self.simult_opt_tsc_sb(sim_data_by_seed)
            
        else:
            if shed_opt == 'grid': #grid of sheds, for each use GN for tscale
                metric_data = self.optim_sb_grid_tsc_gn()
                
            else: #GN on both
                #optimise tscales without anything suprabasal layer related
                metric_data = self.optimise_tscales(sim_data_by_seed, i, exclude = ds.supra_quants)
                metric_data = self.optimise_suprabasal(sim_data_by_seed, i)
        
        self.add_metrics(metric_data)
        
        #Store order parameters
        self.store_order_params(sim_data_by_seed, i)
        
        return
    
           
    def optim_tsc_grid_sb_gn(self):
        return
    
    def optim_sb_grid_tsc_gn(self):
        return    
    
    def set_to_extinct(self, i, extval = np.pi*1e10):
        '''
            Sets flag to indicate extinction
        '''

        simulator.set_to_extinct(self, i)

        for m in self.metrics:
            self.searchspace.at[i,   m.name] = extval
            self.searchspace.at[i, m.name +':tscale'] = extval
            self.searchspace.at[i, m.name +':shed'] = extval     
            
    
    def num_postopt_params(self):
        if len(self.postopt_params) == 0:
            return 1
        else:
            return len(self.postopt_params)

        
    def setup_postopt_params(self, postopt_params):
        '''
            Save post-optimisation parameters as a pandas series
        '''
        if type(postopt_params) == pd.DataFrame:
            self.postopt_params = postopt_params
        else:
            self.postopt_params = pd.DataFrame(product(*postopt_params.values()), columns = postopt_params.keys())
        self.postopt_params = self.postopt_params.applymap(lambda x: round(x, self.dp_round))
        return self.postopt_params

    def get_postopt_params(self, pi):
        if len(self.postopt_params) == 0 or pi > len(self.postopt_params):
            return {}
        else:
            return self.postopt_params.iloc[pi].to_dict()
        
           
           
    def setup_metdata(self, index = -1, maxcost = 1e10, save_sweeps = False):
        '''
            Create metric data series which stores the best cost and best post-run optimisation parameters
        '''
        
        metric_data = pd.Series({m: maxcost for m in self.metric_names}, dtype = 'float64')
        for m in self.metrics:
            metric_data[m.name + ':tscale'] = 1
            metric_data[m.name + ':shed']   = 0
            for popt in self.postopt_params:
                metric_data[m.name + ':' + popt] = -1
            #metric_data[m.name + ':postopt'] = pd.Series()
            if save_sweeps:
                metric_data[m.name + ':sweeps'] = np.ones((len(self.tscalevals), len(self.shedvals), self.num_postopt_params()))*maxcost    
        
        metric_data['index'] = index
        
        return metric_data
    
           
    def compute_metrics(self, metric_data, sim_data, exp_data = None, 
                        ti = None, sbsi = None, pi = None, save_sweeps = False,
                        save_sub_metrics = False):
        '''
            Computes metrics and saves if we have the new best value
        '''
        
        if exp_data is None:
            exp_data is self.exp_data
            
        if sim_data is None: #such as for an invalid tscale value
            return
                        
        for m in self.metrics: #Compute metrics
            cost = m.calc(sim_data, exp_data)
            if cost < metric_data[m.name]:
                metric_data[m.name] = cost
                metric_data[m.name + ':tscale'] =  self.tscalevals[ti]
                metric_data[m.name + ':shed']    = self.shedvals[sbsi]
                for popt in self.postopt_params:
                    metric_data[m.name + ':' + popt] = self.get_postopt_params(pi)[popt]
                #metric_data[m.name + ':postopt'] = pd.Series(self.get_postopt_params(pi))
                
                if save_sub_metrics:
                    best_sub_metrics = m.full_metrics.copy()
                    metric_data[m.name + ':submetrics'] = best_sub_metrics
                    
                    #for sm in best_sub_metrics:
                    #    metric_data[m.name + ':' + sm] = best_sub_metrics[sm]
                
            if save_sweeps:
                #note in indexing convention, frustratingly tscalevals in optimizer and datasets have different sizes
                metric_data[m.name + ':sweeps'][ti, sbsi, pi] = cost
                    
                    
           
    def setup_bootdata(self, maxcost = 1e10):
        '''
            Set-up bootstrap data if we need bootstrap iterations
        '''
        
        if self.nboots > 0:
            row = pd.DataFrame([self.setup_metdata(maxcost = maxcost)])
            boot_data = pd.concat([row] * self.nboots, ignore_index = True) #bootstrapping data
            boot_data['index'] = np.arange(self.nboots)
        else:
            boot_data = None
            
        return boot_data
    
    
    def bootstrap_metrics(self, boot_data, sim_data, ti = None, sbsi = None, pi = None):
        '''
            Computes bootstraps of metrics and saves if we have the new best value
                for given bootstrapped dataset
        '''
        
        if self.nboots == 0:
            return
        
        for b in range(self.nboots):
            b_met_data = boot_data.iloc[b]
            self.compute_metrics(b_met_data, sim_data, exp_data = self.boot_data.subdataset(b),
                        ti = ti, sbsi = sbsi, pi = pi, save_sub_metrics = False)
            boot_data.iloc[b] = b_met_data
            
            
            
    def add_bootstraps(self, metric_data, boot_data):
        '''
            Adds the bootstrapped metrics to the search-space    
        '''

        if self.nboots == 0:
            return
        
        for m in self.metrics:
            metric_data[m.name + ':boot_costs']   = boot_data[m.name].values
            metric_data[m.name + ':boot_tscale']  = boot_data[m.name + ':tscale'].values
            metric_data[m.name + ':boot_shed']    = boot_data[m.name + ':shed'].values
            for popt in self.postopt_params:
                metric_data[m.name + ':boot_' + popt] = boot_data[m.name + ':' + popt].values            
            
    
           
    def calculate_grid_metrics(self, sim_data_by_seed, i, exp_data = None, maxcost = 1e10,
                               tscale_by_quantity = None, tscale_inc_times = None,
                               stop_metric = None, stop_tol = -2, save_sub_metrics = False,
                               use_best_shed = False):
        '''
            Calculates all metrics specified for the simulation data and
                the row number for parameterset, 
                            
            Returns data as a pandas series
            
            Also, implements bootstrapping if required
            
            note in indexing convention, tscalevals in optimizer and datasets may have different sizes
                due to exclusion of values outside of the range, so be careful
            
            
            Extra features:
                To save re-calculating metric for different tscales (when some metrics are expensive) we
                    can specify tscale_by_quantity, which will determine the best tscale that gets the 
                    average value of quantity to match 
                    
                    In this mode use:
                        tscale_inc_times: include only these times in the calculation
                        
                
                Terminate post-optimisation early based on a basal-only metric:
                    stop_metric: metric to skip calculating suprabasal layer if we are above toleranc
                    stop_tol: tolerance for stopping (note this will only work for positive-only metrics)
                        Use a positive value for additivie and negative for multiplicative
 
        '''
        
        if exp_data is None:
            exp_data = self.exp_data
            
        #Set conditions for stopping post-optimisation early
        if stop_metric is not None:
            if stop_metric not in self.metric_names:
                stop_metric = None
            
            if len(self.searchspace) > 0:
                if stop_tol > 0:
                    stop_cost =  np.min(self.searchspace[stop_metric]) + stop_tol
                else:
                    stop_cost = -np.min(self.searchspace[stop_metric]) * stop_tol
                    assert(stop_cost > 0)
            else:
                stop_cost = np.inf
           
        #Object storing the best post-optimisation parameters
        metric_data = self.setup_metdata(i, maxcost, save_sweeps = self.save_sweeps)
        boot_data   = self.setup_bootdata(maxcost)

        if use_best_shed:
            best_shed = self.best_shed(i)
            self.shedvals = [best_shed]
            self.format_shedvals()
            print('Using best shed:', self.shedvals[0])
        
        #Loop over parameters
        for pi in range(self.num_postopt_params()):
            for sbsi in range(len(self.shedvals)):
                shed = self.shedvals[sbsi]
                #Re-run suprabasal layer if we need to, this does nothing if we don't need to
                sim_data_by_seed.rerun_suprabasal(shed, **self.get_postopt_params(pi)) #only works for seeded data
                sim_data_full = sim_data_by_seed.calculate_combined_quantities(full = False)
                
                if tscale_by_quantity is None: #look through only valid time-points
                    tscalevals = sim_data_full.new_to_old_tscidx #range(len(self.tscalevals))
                else:
                    tscalevals = [self.best_tscale_by_quantity(sim_data_full, tscale_by_quantity, inc_times = tscale_inc_times)]
                    if stop_metric is not None: #another early end condition is if tscale is on extremities
                        terminate = (tscalevals[0] == 0) or (tscalevals[0] == len(self.tscalevals)) #termination condition is that tscale is at the maximum
                    else:
                        terminate = False
                
                for ti in tscalevals:
                    sim_data = sim_data_full.data_time_crop(self.tscalevals[ti]) #crop to the simulation times
                    self.compute_metrics(metric_data, sim_data, exp_data, ti, sbsi, pi, 
                                         save_sweeps = self.save_sweeps, 
                                         save_sub_metrics = save_sub_metrics)
                    self.bootstrap_metrics(boot_data, sim_data, ti, sbsi, pi)
                    
                if stop_metric is not None:
                    if (sim_data is not None and metric_data[stop_metric] > stop_cost) or terminate:
                        print('Terminating post-optimisation early for', i)
                        self.add_bootstraps(metric_data, boot_data)
                        return metric_data
        
        self.add_bootstraps(metric_data, boot_data) #save-bootstrapped data
        
        return metric_data
    
    
    
    def best_tscale_by_quantity(self, sim_data_full, quantity = 'basalCS',
                                inc_times = None):
        '''
            Determine tscale based on the re-scaling of the quantity
                This saves re-computing all metrics for every value of t-scale
                
            Takes non-cropped data
            
            inc_times: include only these times in the calculation
        '''
        
        if inc_times is None:
            texp = self.exp_data.times[quantity]
            av_exp = self.exp_data.quantities[quantity]
            se_exp = self.exp_data.se_quants[quantity]
        else:
            texp = self.exp_data.times[quantity][inc_times]
            av_exp = self.exp_data.quantities[quantity][inc_times]
            se_exp = self.exp_data.se_quants[quantity][inc_times]
            exp_nsamples = self.exp_data.nsamples[quantity][inc_times]
            
        tscales = sim_data_full.tscalevals #this has maximum and minimum values excluded
        av_sim = sim_data_full.quantities[quantity]
        
        qcosts = cost_by_tscale(tscales, av_sim, av_exp, se_exp, texp, exp_nsamples,
                       sim_data_full.tshift, sim_data_full.skip)
        assert(np.sum(np.isnan(qcosts)) == 0)   
        assert(np.sum(np.isinf(qcosts)) == 0)
        
        return sim_data_full.new_to_old_tscidx[np.argmin(qcosts)]


    def best_shed(self, p, inc_times = None):
        '''
            Compute the shed value based on the experimental data
        '''

        wam_with_se = lambda v, se: np.array([np.average(v, weights=1/(se**2)), np.sqrt(np.sum(se**2))/len(v)])
        sb_rat = (self.exp_data.raw_data['clone_anal'].av_full_sup_cs/self.exp_data.raw_data['clone_anal'].lcd)

        if inc_times is not None:
            sb_rat = sb_rat[inc_times]

        av_sb_rat = wam_with_se(sb_rat, self.exp_data.raw_data['clone_anal'].se_full_sup_cs/self.exp_data.raw_data['clone_anal'].lcd)

        pset = self.paramset(p)
        tau = pset['tau']

        if self.modelname == 'SD_0D':
            if 'mu' in pset:
                mu = pset['mu']
                if 'useTerm' in pset:
                    useTerm = pset['useTerm']
                else:
                    useTerm = True

                if useTerm:
                    rho = 1/(1 + mu/tau)            
                else:
                    rho = 1/(1 + 2*mu/tau)  
                fD = 1 - rho

            elif 'rho' in pset:
                rho = pset['rho']
                fD = 1 - rho
                if 'useTerm' in pset:
                    useTerm = pset['useTerm']
                else:
                    useTerm = True

                if useTerm:
                    mu = tau*(1-rho)/rho
                else:
                    mu = tau*(1-rho)/rho*2  
            else:
                assert(0)

        elif self.modelname == 'STD_0D':
            if 'mu' in pset:
                mu = pset['mu']
                rhoS = 1/(1+mu/tau)/2
                fD = 1 - 2*rhoS
            elif 'rhoS' in pset:
                rhoS = pset['rhoS']
                fD = 1 - 2*rhoS
                mu = tau*(1-rhoS*2)/(rhoS*2)
            else:
                assert(0)
        
        elif self.modelname == 'SPD_0D':
            if 'q' in pset:
                q = pset['q']
            else:
               q = 1.0

            if 'delta' in pset:
                delta = pset['delta']
            else:
                delta = 0.0

            if 'tauP' in pset:
                tauP = pset['tauP']
            else:
                tauP = tau

            if 'mu' in pset:
                mu = pset['mu']
            elif 'rhoS' in pset:
                rhoS = pset['rhoS']
                mu   = (q*(1 - rhoS)*(-1 + 2*delta)*tau + rhoS*tauP)/(rhoS*(-1 + q*(-1 + 2*delta)))
            else:
                assert(0)
            fD = ((1 + q*(1 - 2*delta))*tau*(1 - rhoS*2))/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*tau*(1 - rhoS*2) + tauP)

        elif self.modelname == 'ST2D_0D':
            if 'q' in pset:
                q = pset['q']
            else:
               q = 1.0

            if 'tauT' in pset:
                tauT = pset['tauT']
            else:
                tauT = tau

            if 'mu' in pset:
                mu = pset['mu']
            elif 'rhoS' in pset:
                rhoS = pset['rhoS']
                mu = ((q-2)*rhoS*tauT + tau*(1-rhoS))/(rhoS*(3-q))
            else:
                assert(0)
            fD = mu*(3-q)/(tau + ((3 - q)*mu + (2 - q)*tauT))

        else:
            print('Best shed not implemented for model:', self.modelname)
            assert(0)

        return av_sb_rat*mu/fD
            


    def add_metrics(self, metric_data):
        '''
            Adds the metrics to the search-space
        '''
        
        i = metric_data['index']
        
        for col_name in metric_data.keys():
            if col_name == 'index':
                continue
            if col_name not in self.searchspace.columns:
                self.searchspace[col_name] = None
            self.searchspace.at[i, col_name] = metric_data[col_name]
        
        for m in self.metrics:
            print('Metric:', m.name, 'Best cost:', metric_data[m.name], 'tscale:', metric_data[m.name + ':tscale'], 
                  'shed:', metric_data[m.name + ':shed'], 
                  list(self.postopt_params.columns),
                  [metric_data[m.name + ':' + popt] for popt in self.postopt_params])
                  #dict(metric_data[m.name + ':postopt']))
            
            
            
    def full_bootstrap_metrics(self, sim_data_by_seed, i, maxcost = 1e10):
        '''
            Calculates best metric for each bootstrap sample
            
            This is the slow version as it requires re-run of suprabasal layer
                for every iteration
        '''
        if len(self.shedvals) > 1 or self.num_postopt_params() > 1:
            print('Warning: bootstrapping metrics requires re-computation of suprabasal layer on every iteration.')
        
        boot_costs   = np.ones( (self.nboots, len(self.metrics)))*maxcost
        boot_tscales = np.ones( (self.nboots, len(self.metrics)))
        boot_sheds   = np.zeros((self.nboots, len(self.metrics)))
        
        for b in range(self.nboots):
            metric_data = self.calculate_grid_metrics(sim_data_by_seed, i, exp_data = self.boot_data.subdataset(b))
            for m in range(len(self.metrics)):
                metname = self.metrics[m].name
                boot_costs[b, m]   = metric_data[metname]
                boot_tscales[b, m] = metric_data[metname + ':tscale']
                boot_sheds[b, m]   = metric_data[metname + ':shed']
            
        for m in range(len(self.metrics)):
            metname = self.metrics[m].name
            self.searchspace.at[i, metname + ':boot_costs']  = boot_costs[:, m]
            self.searchspace.at[i, metname + ':boot_tscale'] = boot_tscales[:, m]
            self.searchspace.at[i, metname + ':boot_shed']   = boot_sheds[:, m]            
            
                               
    
    def simult_opt_tsc_sb(self):
        '''
            Simultaneous optimisation of tscale and suprabasal shedding using
                gradient descent
        '''
        
    
    
    def optimise_suprabasal(self, sim_data_by_seed, i, tscale = None, tsci = None):
        '''
            Post optimising the suprabasal layer 
                default method: Gauss-Newton method
                
            Takes the value of tscale which is specified, else use the 
                one already in the searchspace
        '''
        

        return 
    
    
    def optimise_tscales(self, sim_data_by_seed, i, sbsi = 0, shed = None, exclude = []):
        '''
            Post optimising of tscales
                default method: Gauss-Newton method
                
            sbsi specifies the suprabasal layer
            Exclude are the variables to exclude in the metric
        '''        
        
        
        
        return

    def best_fit(self, metric, constraints = None, return_idx = False):
        '''
            Peruse the fitted search-space and return the parameters, for the best fit
                given a chosen metric
                
            If no metric exists returns None
            
            We can specify constraints, by using a boolean array of length of search-sapce
            
            return_idx = True, returns index, else returns row
            
        '''
        
        if metric in self.metric_names or metric in self.searchspace.columns:
            # Apply constraints if provided
            if constraints is not None:
                filtered_searchspace = self.searchspace[constraints]
            else:
                filtered_searchspace = self.searchspace

            if not filtered_searchspace.empty:
                # Find the index of the minimum value for the specified metric
                min_idx = filtered_searchspace[metric].idxmin()

                # Retrieve the row and its original index
                # Note here we use loc not iloc
                best_row = self.searchspace.loc[min_idx].copy()
                
                best_row['tscale'] = best_row[metric + ':tscale']
                best_row['shed']   = best_row[metric + ':shed']

                if return_idx:
                    return min_idx
                else:
                    return best_row
            else:
                return None
        else:
            return None        

    def get_array_metric_data(self, met, data_name, default_val = 1e10):
        ''' 
            Extract data in a 1D numpy from search-space given metric
        '''            
        #Works only for 1D shapes
        bt = self.searchspace[met + ':' + data_name]
        lens = np.array([len(s) if (s is not None and np.ndim(s) >= 1) else 0 for s in bt])
        if len(np.unique(lens[lens > 0])) == 1:
            bt = [np.ones(lens[lens > 0][0])*default_val if (x is None or (np.ndim(x) == 0)) else x for x in bt]
            return np.stack(bt)
        else:
            print(data_name + 'has inconsistent shapes')
            minlens = np.min(lens[lens > 0])
            bt = [np.ones(minlens)*default_val if (x is None or (np.ndim(x) == 0)) else x[:minlens] for x in bt]
            return np.stack(bt)
            #return None
        
        
    def boot_costs(self, met, maxcost = 1e10):
        ''' 
            Extract boots from search-space given metric
        ''' 
        
        return self.get_array_metric_data(met, 'boot_costs', maxcost)
                  
        
    def boot_tscales(self, met, default_val = 1e10):
        ''' 
            Extract best tscale for bootstrapped search-space given metric
        '''            

        return self.get_array_metric_data(met, 'boot_tscale', default_val)


    def boot_shed(self, met, default_val = 1e10):
        ''' 
            Extract best shd for bootstrapped search-space given metric
        '''            

        return self.get_array_metric_data(met, 'boot_shed', default_val)        
    

    
    def boot_stats(self, met, timelike = None, include_values = False, constraints = None):
        ''' 
            Extract bootstrapped statistics from search-space given metric
            
            timelike variables are re-scaled by the best tscale
        '''
               
        if constraints is not None:
            filtered_searchspace = self.searchspace[constraints]
        else:
            filtered_searchspace = self.searchspace               
        
        if timelike is None:
            timelike = ['tau']
            
        
        bst = pd.DataFrame(columns = ['Variable', 'Best', 'Mean', 'Median', 
                                      'Std', 'Min', 'Max', '95% CI l', '95% CI u', 
                                      '68% CI l', '68% CI u', 'Values'])
        
        param_search = list(set(self.freeParamVars + timelike + [met +':tscale'] + [met +':shed']))
        
        param_search += [met + ':' + c for c in list(self.postopt_params.columns)]


        if 'tau' in self.allParamVars and 'r' in self.allParamVars:
            param_search.append('τ/r')
            filtered_searchspace['τ/r'] = filtered_searchspace['tau']/filtered_searchspace['r']
            timelike.append('τ/r')

        if 'rho' in self.allParamVars:
            param_search.append('μ')
            filtered_searchspace['μ'] = filtered_searchspace[met +':tscale']*(1 - filtered_searchspace['rho'])/filtered_searchspace['rho']

        if (filtered_searchspace[met +':shed'] < 0).any():
            filtered_searchspace['shed'] = filtered_searchspace[met +':shed']*filtered_searchspace[met +':tscale']
            param_search.append('shed')

        for p in param_search:
            boot_vals = filtered_searchspace[p][np.argmin(self.boot_costs(met),0)]
            best_val  = filtered_searchspace[p][np.argmin(filtered_searchspace[met])]
            if p in timelike:
                boot_vals /= self.boot_tscales(met)[np.argmin(self.boot_costs(met),0)].diagonal()
                best_val  /= filtered_searchspace[met + ':tscale'][np.argmin(filtered_searchspace[met])]      
            stats_p = pd.Series({'Variable': p,
                                 'Best': best_val,
                                 'Mean': np.mean(boot_vals),
                                 'Median': np.median(boot_vals),
                                 'Std': np.std(boot_vals),
                                 'Min': np.min(boot_vals),
                                 'Max': np.max(boot_vals), 
                                 '95% CI l': np.percentile(boot_vals, 2.5),
                                 '95% CI u': np.percentile(boot_vals, 97.5),
                                 '68% CI l': np.percentile(boot_vals, 16),
                                 '68% CI u': np.percentile(boot_vals, 85),
                                 'Values': boot_vals})
            bst = pd.concat([bst, stats_p.to_frame().T], ignore_index = True,)
        

        bst.index = bst.Variable.values
        bst = bst.drop('Variable', axis = 1)
        
        if not include_values:
            bst = bst.drop('Values', axis = 1)
            
        return bst
            
            


    def sweeps(self, met, maxcost = 1e10):
        ''' 
            Extract sweeps from search-space given metric
        '''
        
        if met + ':sweeps' in self.searchspace:
            sweeps = self.searchspace[met + ':sweeps']
            shapes = [s.shape for s in sweeps if (s is not None and np.ndim(s) >= 1)]
            
            if len(set(shapes)) == 1:
                # Shape of the arrays to replace None
                consistent_shape = shapes[0]

                # Replace None with an array of 1e10 of the consistent shape
                sw = [np.full(consistent_shape, maxcost) if (x is None or np.ndim(x) == 0) else x for x in sweeps]
                sw = np.stack(sw)
                sw[np.isnan(sw)] = 1e10
                
                return sw
            else:
                print('Sweeps have inconsistent shapes')
                return None            
        else:
            return None
        
        
        

    def param_sweep(self, indepvars, met, fix_params, tsci = None, sbsi = None):
        ''' 
            Gets sweep of a metric over a parameter space
                for specified independent variables
                
            Inputs:
                indepvars: list of independent variables to get params for
                met: metric to use
                fix_params: fixing non-varying parameters
                tsci: index of tscale to use, if not varying over
                    if None, we use best, otherwise use that index
                sbsi: index of shed to use, if not varying over
                    if None, we use best, otherwise use that index
                
            Returns:
                sub_sweep: multidimensional array of cost over the sweep
                indepvals: list of 1D arrays of independent variable values
                
                
            If we don't vary have self.sweep then we must use the best values as 
                specified in search-space
        '''
        
        
        # Extract independent variables to plot with
        if 'tscale' in indepvars:
            indepvars.remove('tscale')
            use_tscale = True
        else:
            use_tscale = False
            
        if 'shed' in indepvars:
            indepvars.remove('shed')
            use_shed = True
        else:   
            use_shed = False

        if tsci is None and use_tscale == False:
            print('Using best tscale for each parameter set')
        
        if sbsi is None and use_shed == False:
            print('Using best shed for each parameter set')
            
            
        # Set-up fixed parameters to filter
        fix_params = {**self.fixedParams, **fix_params}

        # Get metric index
        if met in self.metric_names:
            midx = self.metric_names.index(met)
        else:
            print('Metric not found, using', self.metric_names[0])
            midx = 0
            
        subdf = self.fullparams.query(dict_to_dfquery(fix_params))#[indepvars]
        assert(subdf[indepvars].duplicated().any() == False)
        assert(subdf.empty == False)
        
        '''
        #Indices for 1D, now hav a general approach
        vals = subdf.values
        idxs = np.array(subdf.index)
        idxs = idxs[np.argsort(vals)]
        vals = vals[idxs]
        '''
        
        idxs, indepvals = df_to_ndarray(subdf, indepvars)
        
        missing = idxs < 0
        
        if use_tscale:
            indepvals.append(self.tscalevals)
            indepvars.append('tscale')
            
            if use_shed:
                sub_sweep = self.sweeps(met)[idxs, :, :] 
                sub_sweep[missing, :, :] = np.nan
            else:
                if sbsi is None:
                    sub_sweep = np.min(self.sweeps(met)[idxs, :, :], -1)
                else:
                    sub_sweep = self.sweeps(met)[idxs, :, sbsi]
                sub_sweep[missing, :] = np.nan
        else:
            if use_shed:
                if tsci is None:
                    sub_sweep = np.min(self.sweeps(met)[idxs, :, :], -2)
                else:
                    sub_sweep = self.sweeps(met)[idxs, tsci, :]
                sub_sweep[missing, :] = np.nan
            else:
                if tsci is None:
                    if sbsi is None: #best over tscale and sbsi
                        sub_sweep = self.searchspace[met].values[idxs]
                    else:
                        sub_sweep = np.min(self.sweeps(met)[idxs, :, sbsi], -1)
                else:
                    if sbsi is None:
                        sub_sweep = np.min(self.sweeps(met)[idxs, tsci, :], -1)
                    else:
                        sub_sweep = self.sweeps(met)[idxs, tsci, sbsi]
                sub_sweep[missing] = np.nan
        
        if use_shed:
            indepvals.append(self.shedvals)
            indepvars.append('shed')
        
        if len(idxs) == 1:
            sub_sweep = sub_sweep.flatten()
        
        return sub_sweep, indepvals


    def plot_sweeps(self, indepvars, met, fix_params, tsci = None, sbsi = None,
                    good_thresh  = None, mark_best = False, maxcost = 1e10,
                    cbarlabel = 'cost'):
        '''
            Plots the independent variable parameter sweeps
            
            Inputs:
                indepvars: list of independent variables to get params for
                met: metric to use
                fix_params: fixing non-varying parameters
                tsci: index of tscale to use, if not varying over
                    if None, we use best, otherwise use that index
                sbsi: index of shed to use, if not varying over
                    if None, we use best, otherwise use that index
                    
                good_thresh: threshold above the best fit marked as a good fit, 
                    if None then we use only plot the best fit
                    Good fits are marked with a dot
                            
        '''
        
        assert(len(indepvars) <= 2)
        
        sub_sweep, indepvals = self.param_sweep(indepvars, met, fix_params, tsci, sbsi)
        print('Min cost:', np.min(sub_sweep))
        sub_sweep[sub_sweep >= 1e10] = np.nan
        
        if len(indepvars) == 1:
            plt.plot(indepvals[0], sub_sweep, 'o')
            plt.xlabel(indepvars[0])
            plt.ylabel(met)
            
        elif len(indepvars) == 2:
            dx = (indepvals[0][-1] - indepvals[0][0])/(len(indepvals[0]) - 1)
            dy = (indepvals[1][-1] - indepvals[1][0])/(len(indepvals[1]) - 1)
            plt.imshow(sub_sweep.transpose(), extent = [indepvals[0][0] - dx/2, indepvals[0][-1] + dx/2, 
                        indepvals[1][0] - dy/2, indepvals[1][-1] + dy/2], origin = 'lower', aspect = 'auto')
            
            plt.colorbar(label = cbarlabel)
            if len(indepvals[0]) <= 10:
                plt.xticks(indepvals[0])
            else:
                N = int(np.ceil(len(indepvals[0])/4))
                xticklocs = indepvals[0][0] + dx*np.arange(0,len(indepvals[0] + 1),N)
                plt.xticks(xticklocs, round_sf(indepvals[0][::N],3))
                
            if len(indepvals[1]) <= 10:
                plt.yticks(indepvals[1])
            else:
                N = int(np.ceil(len(indepvals[1])/4))
                yticklocs = indepvals[1][0] + dy*np.arange(0,len(indepvals[1] + 1),N)
                plt.yticks(yticklocs, round_sf(indepvals[1][::N],3))                
                
            #Plot the best fit
            min_index = np.unravel_index(np.nanargmin(sub_sweep), sub_sweep.shape)            
            if good_thresh is not None:
                good = np.argwhere(sub_sweep < (good_thresh + np.min(sub_sweep[~np.isnan(sub_sweep)])))
                plt.plot(indepvals[0][0] + dx*(good[:,0]),
                     indepvals[1][0] + dy*(good[:,1]), 'k.')
                
                #plt.plot(indepvals[0][good[:,0]], indepvals[1][good[:,1]], 'k.')
            #plt.plot(indepvals[0][min_index[0]], indepvals[1][min_index[1]], 'x', color = 'red')
            if mark_best:
                plt.plot(indepvals[0][0] + dx*(min_index[0]), indepvals[1][0] + dy*(min_index[1]), 'x', color = 'red')
            
            if indepvars[0] == 'tscale':
                plt.ylabel(r'$\tau$')
            else:
                plt.xlabel(indepvars[0])
                
            if indepvars[1] == 'tscale':
                plt.ylabel(r'$\tau$')
            else:
                plt.ylabel(indepvars[1])
        
        #plt.show()


   
   
def cost_by_tscale(tscale, av_sim, av_exp, se_exp, texp, exp_nsamples = 0, tshift = 0, skip = 50):
    '''
        Cost function for a given tscale
    '''
    include = (se_exp > 0) & (av_exp > 0)
    if np.ndim(tscale) == 0:
        return np.sum((av_sim[((texp - tshift)*tscale/skip).astype('int32')] - av_exp)**2/(se_exp**2 + (se_exp == 0)/exp_nsamples))
    else:
        return np.sum((av_sim[((texp[:,None] - tshift)*np.array(tscale)[None,:]/skip).astype('int32')] - av_exp[:,None])**2/(se_exp**2 + (se_exp == 0)/exp_nsamples)[:,None], axis = 0)

   
#Useful functions:

def pre_defined_metrics(metric_name):
    #Takes a metric name
    return

def dict_to_dfquery(d):
    q = ''
    for v in d.keys():
        if type(d[v]) is str:
            q += ' & `' + v + '` == "' + str(d[v]) + '"'
        else:
            q += ' & ' + v + ' == ' + str(d[v]) 
    return q[3:]

def actualseed(i, s, maxseed = 1000):
    '''
        Way of generating seed that allows better random numbers
    '''
    return maxseed*i + s

def round_sf(arr, sf):
    #rounds numpy array or number to a certain amount of significant digits  
    #Handles 0, positive or negative numbers  
    if np.ndim(arr) == 0:
        if arr == 0.0:
            return 0.0
        flag0 = 0.0
    else:
        flag0 = arr == 0
        #arr[arr == 0] += 1e-10
    scale_factor = 10 ** (sf - 1 - np.floor(np.log10(np.abs(arr + flag0*1e-10))))
    return np.around(arr * scale_factor) / scale_factor - flag0*1e-10


def df_to_ndarray(df, indepvars):
    ''' 
        Extracts indices to convert a data-frame to a numpy ndarray
            with a sweep over these parameters
            
        Returns:
            idx_array: ndarray with indices for each independent variable
            unique_values: list of unique values for each independent variable
            
        -1: value not present in data-frame
    
    '''
    
    if len(indepvars) == 0:
        return np.array(df.index), []
    
    # Ensure indepvars is a list of column names
    assert isinstance(indepvars, list) and len(indepvars) > 0

    # Get unique values and sort them for each independent variable
    unique_values = [np.sort(df[var].unique()) for var in indepvars]

    # Create a mapping from value to index for each independent variable
    mappings = [{v: i for i, v in enumerate(unique)} for unique in unique_values]

    # Create the shape for the ndarray
    shape = [len(unique) for unique in unique_values]

    # Initialize the ndarray with a placeholder value (e.g., -1)
    idx_array = -1 * np.ones(shape, dtype=int)

    # Fill the array with indices
    for idx, row in df.iterrows():
        indices = [mappings[i][row[var]] for i, var in enumerate(indepvars)]
        idx_array[tuple(indices)] = idx

    return idx_array, unique_values


'''
class sim_anneal(optimizer):
    def __init__(self, model, params):
        optimizer.__init__(model, params)
        
        
class gd(optimizer):
    def __init__(self, model, params, datafile, compare, metrics, seed=None):
        super().__init__(model, params, datafile, compare, metrics, seed)


'''

def convert_to_int(value):
    if np.ndim(value) > 0:
        return value.astype(int)
    else:
        return int(value)



#Import
from sim_anneal import *

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def is_bool(s):
    if s.lower() in ['true', 'false']:
        return True
    else:
        return False