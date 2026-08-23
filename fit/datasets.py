import os
import pickle
import numpy as np
import copy
import analysis
import tissue
from ds_plot import *
import clones
import shape_analysis as sa

'''
    Note: we can can handle sub-population quantities, by specifying a quantity
        as 'popname.X', but the code must be set-up to compute the population 
        quantity directly

'''

zero_arr = np.zeros(1, dtype = int)


#Preamble: defines the types of data outlined below

#Data types currently implemented
categories = ['basalCS', 'totalCS', 'supraCS', 'density', 'divrate', 'pers', 
              'lcd', 'death', 'T1s', 'MSD', 'shape', 'jointCS', 'tranCS',  'popjointCS',
              'ctjointCS', 'btranCS', 'fate_reff', 'area', 'xfrac',
              'fullbasalCS', 'fullsupraCS', 'fulltotalCS', 'fulljointCS']

shortname = {'basalCS': 'bas', 'totalCS': 'tot', 'supraCS': 'sup', 'density': 'dens', 
             'divrate': 'dr', 'pers': 'pers', 'lcd': 'lcd', 'death': 'dth', 
             'jointCS': 'jcs', 'tranCS': 'tcs', 'popjointCS': 'pjcs', 'ctjointCS': 'ctjcs', 
             'btcs': 'btranCS', 'fate_reff': 'fate', 'xfrac': 'xf', 
             'fullbasalCS': 'fbas', 'fullsupraCS': 'fsup', 'fulltotalCS': 'ftot', 'fulljointCS': 'fjcs',
             'shape': 'shape', 'area': 'area'}


#Whether the quantities are skipped in simulation
skipped = {'basalCS': True, 'totalCS': True, 'supraCS': True, 
           'density': False, 'divrate': False, 'pers': True, 'lcd': True, 'death': True,
           'T1s': True, 'MSD': True, 'shape': True, 'jcs': True, 'tcs': True, 'ctjcs': True,
           'fate_reff': False, 'xfrac': False, 'area': True}

#Categories which affected by cells in the suprabasal layer
supra_cats = ['totalCS', 'supraCS', 'jointCS', 'tranCS', 'fulljointCS',  'fullsupraCS', 'fulltotalCS']
clonal_cats = ['basalCS', 'totalCS', 'supraCS', 'lcd', 'pers', 'jointCS', 'tranCS', 'btranCS',
               'popjointCS', 'ctjointCS', 'xfrac', 'fullbasalCS', 'fullsupraCS', 'fulltotalCS',
               'fulljointCS'] #marking by populations
    

class dataset():
    '''
        Dataset type stores:
            timeless: is a scalar or vector quantity that is fixed over time, has a dimension
            quantity: is a quantity that varies over time
               dists: is a probability distribution, stored as a pdf form
             For distributions, we also store the number of samples 
                and dist_bins, if dist_bins = None then we 
                    assume the bins are 0,1,2,...
                To store continuous data len(dist_bins) = len(dist) + 1
                To store discrete data len(dist_bins) = len(dist)
             
        Data can be seeded or non-seeded, seeded data has the 0-th index corresponding to the seed
            or index of the biological replicate
    '''
    
    def __init__(self, toCalculate = [], max_pdf_bin = None, compute_kwargs = None):
        '''
            Create base datasets, quantities, distributions and times associated with indices
            we also have seeds which allows the instantiation of simulations or biological 
            replicates, seeds
            
            Stores extra-dimensions and names which might be different for different variables
            Defaults are for tscales and suprabasal
            
            compute_kwargs allows passing in extra compute_kwargs as they arise
                compute_kwargs is a dict associated with a particular type of data, 
                    e.g. {'cttranCS': {'ctypes': [1,2]}}
                or can be associated with all data for a particular population:
                    To specify for all population use 'all'
                    e.g. {'Dlike': {'containsBasal': False}, 'all': {'containsBasal': True}}
                If we don't specify for a particular population we take the default for 'all'

            
        '''
        
        self.toCalculate = toCalculate
        self.subpops = self.what_subpops() #sub-populations to compute statistics for
        
        self.check_valid_cats()

        self.timeless   = {} #timeless quantities
        self.quantities = {} #quantities at each time-point
        self.dists = {} #probability distributions,
        self.dist_bins = {}

        self.raw_data = {}  #allows storage of custom data
        
        #custom_quants are calculated at the stage that data is unseeded and collapsed
        self.custom_quants = {} #allows storage of quantities calculated from raw data
        
        self.nsamples = {} #number of samples at each time-point
        self.times = {}
        self.nseeds = 0
        self.added_seeds = set()
        self.type = 'ds'
        self.se_quants  = {}   
    
        
        if max_pdf_bin is None:
            max_pdf_bin = {}
        
        self._max_pdf_bin = max_pdf_bin
        self.char_time = 1.0 #characteristic time
        
        #Flags: important  for behaviour of how it operates
        self.collapsed = True #flag indicating whether we have a grid over additional dimensions
        self.seeded = False
        self.alltimes = False #For simulations this flag allows different behaviour depending on time-reducing
        
        #Extra compute_kwargs
        self.setup_compute_kwargs(compute_kwargs)



    def setup_compute_kwargs(self, compute_kwargs = None):
        '''
            Set-up compute kwargs
        '''

        if compute_kwargs is None:
            compute_kwargs = {}        

        self._compute_kwargs = compute_kwargs
        
        if 'all' not in compute_kwargs:
            compute_kwargs['all'] = {}

        #Takes the defaults as defaults            
        compute_kwargs['all'] = {**self.compute_kwargs_defaults, **compute_kwargs['all']}


    @property
    def compute_kwargs_defaults(self):
        '''
            Default values of compute_kwargs
        '''

        return {'containsBasal': True}
         


    def compute_kwargs(self, x = 'all', p = None):
        '''
            x is either a population, or specific type of data, or 'all'

            p is the particular parameter

            If p is None, returns the dictionary
            Else returns the particular parameter

            We default to 'all' if not supplied
        '''

        #Default to 'all'
        if x is None:
            x = 'all'

        if p is None: #no parameter provided
            if x in self._compute_kwargs:
                return self._compute_kwargs[x]
            else:
                return self._compute_kwargs['all']
        else:
            if x in self._compute_kwargs and p in self._compute_kwargs[x]:
                return self._compute_kwargs[x][p]
            elif p in self._compute_kwargs['all']:
                return self._compute_kwargs['all'][p]
            else:
                print(f'{p} is not in compute_kwargs')
                return None



    def what_subpops(self):
        '''
            Extract which sub-population data to compute
        '''

        self.subpops = set([tc.split('.')[0] for tc in self.toCalculate if '.' in tc])
        self.subpops.add(None)
        return self.subpops


    def check_valid_cats(self):
        for tc in self.toCalculate:
            if '.' not in tc:
                assert(tc in categories), 'Invalid quantity specified ' + tc
            else:
                assert(tc.split('.')[1] in categories), 'Invalid quantity specified ' + tc

        #Calculate basal clone size as a default
        if self.has_clonal_quants() and 'basalCS' not in self.toCalculate:
            self.toCalculate.append('basalCS')        
        

    @property
    def max_pdf_bin(self):
        return self._max_pdf_bin


    def get_max_pdf_bin(self, d):
        self.fix_max_pdf_bin()
        if d not in self._max_pdf_bin:
            self._max_pdf_bin[d] = None
        return self._max_pdf_bin[d]

    
    def fix_max_pdf_bin(self, d = None):
        '''
            Fix the max_pdf_bin based on current values
        '''

        if d is None:
            for d in self.dists:
                self._max_pdf_bin[d] = self.dists[d].shape[1 + self.seeded] - 1
        else:
            self._max_pdf_bin[d] = self.dists[d].shape[1 + self.seeded] - 1

        
    def sample(self, d, t = 0, seed = None):
        '''
            Returns samples for a given distribution at a given time-point
                indexed by t for tscale-idx 0

            Defaults to: returning combining data across all seeds
            Alternatively can specify a seed
            
            Does not work with data with extra-dimensions
            
        '''
                            
        if self.seeded is True:
            if seed is None: #use combined over sims
                N = np.sum(self.nsamples[d][:,t],0)
                hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t], axis=0) * N).astype(int)
            else:
                N = self.nsamples[d][seed,t]
                hist_counts = (self.dists[d][seed, t, :]*N).astype(int)         
        else:
            N = self.nsamples[d][t]
            hist_counts = (self.dists[d][t, :]*N).astype(int)
            
        return np.repeat(np.arange(len(hist_counts)), hist_counts)
        
        
    
    def timeq(self, q):
        if q in self.times:
            if self.times[q] is None:
                return zero_arr
            else:
                return self.times[q]
        else:
            if '.' in q and q.split('.')[1] in self.times:
                return self.times[q.split('.')[1]]
            elif self.has_clonal_quants():
                return self.times['basalCS']
            else:
                print('No valid times')
                assert(0)
    
    def timeless_quant(self, q):
        '''
            Set a quantity to be timeless
        '''

        self.times[q] = None

    def is_timeless(self, q):
        return q in self.timeless.keys() or (self.timeq(q) == zero_arr).all()

    def all_quantities(self):
        return {**self.quantities, **self.dists, **self.timeless} #, **self.samples}
    
    def copy(self):
        return copy.deepcopy(self)      

    def shallowcopy(self) :
        return copy.copy(self)
    
    def load_data_file(self, filename, prefix = 'ctr', suffix = ''):
        self.filename   = filename.split('/')[-1].split('.pkl')[0]
        self.folder = os.path.dirname(filename)  
        if self.folder == '':
            self.folder = '.'
        data_full = load_object(self.folder + '/' + self.filename + '.pkl')
        return data_full


    def max_time(self):
        ''' 
            Returns maximum time
        '''
        
        mt = -np.inf
        for q in self.times.keys():
            mt1 = np.max(self.timeq(q))
            if mt1 > mt:
                mt = mt1
        return mt


    def min_inc(self):
        ''' 
            Returns the minimum time increment used in the data
        '''
        
        mt = np.inf
        for q in self.times.keys():
            if self.ntimes_by_quantity(q) > 1:
                mt1 = np.min(self.timeq(q)[1:] - self.timeq(q)[:-1])
                if mt1 < mt:
                    mt = mt1
        return mt


    def reshape_pdf(self, d, max_bin = None, zeropad = 2, update = True):
        '''
            Re-shapes an individual pdf, by following reshape_pdfs
        '''
        
        newdists = self.reshape_pdfs(zeropad, max_bin = {d: max_bin}, dists_to_reshape = [d], update = update)
        return newdists[d]
        

    def reshape_pdfs(self, zeropad = 2, max_bin = None, dists_to_reshape = None, update = True):
        '''
            Re-shapes pdfs to ensure they have consistent sizes to compare with simulation
            
            The default is re-shape so the bins go up to maximum size + zeropad 
            
            Then simulation clones larger than experimental clones can go in the largest bin
            
            Reshapes pdfs to zeropad, else can force to max_bin
            
            Assumes the dataset is not seeded
            
            If dists_to_reshape is provided then we re-shape specific dists, 
                otherwise we re-shape all
            
            If update = True, we save the data
            
        '''
        
        if dists_to_reshape is None:
            dists_to_reshape = self.dists.keys()
            
        s = int(self.seeded) + 1
        
        newdists = {}
        
        for d in dists_to_reshape:
            if max_bin is not None and d in max_bin.keys():
                max_pdf_bin  = max_bin[d]  
            else:
                max_pdf_bin = np.nonzero(np.sum(self.dists[d],0))[0][-1] + zeropad
            
            dist = self.dists[d]
            if np.shape(dist)[s] >= (max_pdf_bin + 1):
                #Put the tail into the last datapoint
                dist[..., max_pdf_bin] = np.sum(dist[..., max_pdf_bin:], -1)
                dist = dist[...,:(max_pdf_bin + 1)]           
                        
            else:
                if self.seeded:
                    newpdf = np.zeros((self.nseeds, self.ntimes_by_quantity(d), max_pdf_bin + 1))
                else:
                    newpdf = np.zeros((self.ntimes_by_quantity(d), max_pdf_bin + 1))                    
                newpdf[...,:np.shape(dist)[s]] = dist
                dist = newpdf
            
            newdists[d] = dist
            
            if update is True:
                self._max_pdf_bin[d] = max_pdf_bin
                self.dists[d] = dist
                            
        return newdists
 
           
    def add_seeded_dataset(self, data1, combine_custom = True):
        '''
            Merge data-file with self, replacing the seeds from self with those from data1
        '''
        assert(self.seeded == True)
        
        if hasattr(self, 'extinct'):
            self.extinct = self.extinct or data1.extinct
        
        for s in data1.added_seeds:
            for q in self.timeless.keys():
                if data1.seeded is True:
                    self.timeless[q][s,...] = data1.timeless[q][s,...]
                else:
                    self.timeless[q][s,...] = data1.timeless[q]

            for q in self.quantities.keys():
                if data1.seeded is True:
                    self.quantities[q][s,...] = data1.quantities[q][s,...]
                else:
                    self.quantities[q][s,...] = data1.quantities[q]                   
                    
            for d in self.dists.keys():
                if data1.seeded is True:    
                    assert(self.dists[d].shape[1] == data1.dists[d].shape[1])
                    assert(self.dists[d].shape[2] == data1.dists[d].shape[2])                                
                    self.dists[d][s,...] = data1.dists[d][s,...] 
                    self.nsamples[d][s,...] = data1.nsamples[d][s,...]
                else:
                    assert self.dists[d].shape[1] == data1.dists[d].shape[0] , d
                    assert self.dists[d].shape[2] == data1.dists[d].shape[1] , d
                    self.dists[d][s,...] = data1.dists[d]
                    self.nsamples[d][s,...] = data1.nsamples[d]                   
                   
            if hasattr(self, 'pop_quantities'):
                for q in self.pop_quantities.keys():
                    if data1.seeded is True:                                
                        self.pop_quantities[q][s,...] = data1.pop_quantities[q][s,...]
                    else:
                        self.pop_quantities[q][s,...] = data1.pop_quantities[q]
            
            if hasattr(self, 'pop_dists'):
                for d in self.pop_dists.keys():
                    if data1.seeded is True:                                                
                        self.pop_dists[d][s,...] = data1.pop_dists[d][s,...]
                        self.pop_nsamples[d][s,...] = data1.nsamples[d][s,...]
                    else:
                        self.pop_dists[d][s,...] = data1.pop_dists[d]  
                        self.pop_nsamples[d][s,...] = data1.pop_nsamples[d]                  

            if hasattr(self, 'custom_quants'):
                for q in self.custom_quants.keys():
                    if data1.seeded is True:                                
                        print('Not implemented')
                        assert(0)
                    else:
                        if np.ndim(data1.custom_quants[q]) == 3:
                            #This is for a joint-distribution
                            self.custom_quants[q][s,:,:data1.custom_quants[q].shape[1],:data1.custom_quants[q].shape[2]] = data1.custom_quants[q]
                            self.nsamples[q][s,...] = data1.nsamples[q]
                        else:
                            print('Not implemented')
                            assert(0)

            if combine_custom is True:
                self.combine_seeded_custom_data(data1)
                    
        return self
                
    
    def combine_seeded_custom_data(self, data1):
        ''' 
            Combine the raw data:
                need to specify how to combine data by data
        '''
        
        if 'liveCS' in self.raw_data:
            self.raw_data['liveCS'] = self.raw_data['liveCS'] + data1.raw_data['liveCS']
            
        if 'clone_anal' in self.raw_data:
            self.raw_data['clone_anal'] = self.raw_data['clone_anal'] + data1.raw_data['clone_anal']

                
    def ntimes_by_quantity(self, quantity):
            '''
                Extracts the numbers of times that are saved
            '''
 
            return len(self.timeq(quantity))  
        
       
    def seed_dataset(self, nseeds = 1):
        ''' 
            Takes existing datasets and makes quantities and distributions seeded
                Useful for bootstrapping, etc.
                
            Note this will alter the current dataset
        '''
        
        if not self.seeded:
            self.seeded = True
            self.nseeds = nseeds
            
            for q in self.timeless.keys():
                self.timeless[q] = np.tile(np.array(self.timeless[q]), (nseeds, *self.timeless[q].ndim *(1,)))

            for q in self.quantities.keys():
                self.quantities[q] = np.tile(np.array(self.quantities[q]), (nseeds, *self.quantities[q].ndim *(1,)))
                
            for d in self.dists.keys():
                self.dists[d] = np.tile(np.array(self.dists[d]), (nseeds, *self.dists[d].ndim *(1,)))
                self.nsamples[d] = np.tile(np.array(self.nsamples[d]), (nseeds, *self.nsamples[d].ndim *(1,)))
                
            if hasattr(self, 'pop_quantities'):                
                for q in self.pop_quantities.keys():
                    self.pop_quantities[q] = np.tile(np.array(self.pop_quantities[q]), (nseeds, *self.pop_quantities[q].ndim *(1,)))
            
            if hasattr(self, 'pop_dists'):
                for d in self.pop_dists.keys():
                    self.pop_dists[d] = np.tile(np.array(self.pop_dists[d]), (nseeds, *self.pop_dists[d].ndim *(1,)))
                    self.pop_nsamples[d] = np.tile(np.array(self.pop_nsamples[d]), (nseeds, *self.pop_nsamples[d].ndim *(1,)))
     
            if hasattr(self, 'custom_quants'):
                for d in self.custom_quants.keys():
                    self.custom_quants[d] = np.tile(np.array(self.custom_quants[d]), (nseeds, *self.custom_quants[d].ndim *(1,)))
                    self.nsamples[d] = np.tile(np.array(self.nsamples[d]), (nseeds, *self.nsamples[d].ndim *(1,)))




    def bootstrap(self, nboots = 100, seed = None):
        '''
            Returns a bootstrapped dataset:
                Currently only works for basal, supra and total clonal size
        '''
        
        if seed is not None:
            np.random.seed(seed)
            print('Setting seed to ' + str(seed) + ' for bootstrapping')

        assert(self.seeded is False)
        assert(self.collapsed is True)
                
        boot_data = self.copy()
        curr_data = self.copy()
        boot_data.seed_dataset(nboots)

        for b in range(nboots):
            curr_data.resample_clonal_data()
            curr_data.added_seeds = {b}
            boot_data.add_seeded_dataset(curr_data, combine_custom = False)
        
        return boot_data


    def resample_clonal_data(self):
        ''' 
            Produces a single re-sampled clonal dataset
        '''

        assert(self.seeded is False)
        assert(self.collapsed is True)
        
        if self.has_clonal_quants():
            sample = np.random.randint(0, self.raw_data['liveCS'].shape[0], size = self.raw_data['liveCS'].shape[0])
            livesize = self.raw_data['liveCS'][sample,:,:]

            for subpop in self.subpops:
                #calculate clones
                self.compute_clonal_quantities(livesize, subpop, full = True)      


    def bootstrap_old(self, nboots = 100, seed = None):
        '''
            Returns a bootstrapped dataset:
                Currently only works for basal, supra and total clonal size
        '''
        
        if seed is not None:
            np.random.seed(seed)
            print('Setting seed to ' + str(seed) + ' for bootstrapping')

        assert(self.seeded is False)
        assert(self.collapsed is True)
                
        boot_data = self.copy()
        boot_data.seed_dataset(nboots)
        
        if 'basalCS' in self.toCalculate:
            for t in range(len(self.times['basalCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('basalCS') + 1), size=(nboots, self.nsamples['basalCS'][t]), p = self.dists['basalCS'][t,:])
                boot_data.quantities['basalCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['basalCS'][b, t, :] = getPdf(sample[b, :], self.get_max_pdf_bin('basalCS'))

        if 'totalCS' in self.toCalculate:
            for t in range(len(self.times['totalCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('totalCS') + 1), size=(nboots, self.nsamples['totalCS'][t]), p = self.dists['totalCS'][t,:])
                boot_data.quantities['totalCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['totalCS'][b, t, :] = getPdf(sample[b,:], self.get_max_pdf_bin('totalCS'))
                    
        if 'supraCS' in self.toCalculate:
            for t in range(len(self.times['supraCS'])):
                sample = np.random.choice(np.arange(self.get_max_pdf_bin('supraCS') + 1), size=(nboots, self.nsamples['supraCS'][t]), p = self.dists['supraCS'][t,:])
                boot_data.quantities['supraCS'][:, t] = np.mean(sample, 1)
                for b in range(nboots):
                    boot_data.dists['supraCS'][b, t, :] = getPdf(sample[b,:], self.get_max_pdf_bin('supraCS'))

        return boot_data    
    
    
    
    def bootstrap_SEs(self, d, exp_nsamples, nboots = 100, seed = None, sedists = False):
        '''
            Bootstraps standard errors of distribution, 
                this only is valid for unseeded data
                
                saves to standard error of quantity 
                sedists = True: returns the standard error of the distribution
                    else returns the standard error of the quantity
                
            Inputs:
                d: distribution to bootstrap
                exp_nsamples: number of samples to use from distribution, use the experimental ons
                nboots: number of bootstrap samples to take
                seed: seed for reproducibility of bootstrapping
        '''
        
        if exp_nsamples is None:
            exp_nsamples = self.nsamples[d]
            print('No samples provided, using simulated')
        
        if seed is not None:
            np.random.seed(seed)
            print('Setting seed to ' + str(seed) + ' for bootstrapping')        
        
        assert(self.seeded is False)
        assert(self.collapsed is True)
        
        if d not in self.dists: #do not if not a distribution
            return
        
        se_dists = np.zeros((self.ntimes_by_quantity(d), self.get_max_pdf_bin(d) + 1))
        assert(np.shape(se_dists) == np.shape(self.dists[d])), 'dists and se_dists have same shapes'
        self.se_quants[d] = np.zeros(self.ntimes_by_quantity(d))
        if d in self.quantities:
            assert(np.shape(self.se_quants[d]) == np.shape(self.quantities[d])), 'quantities and se_quants have same shapes'
                
        for t in range(self.ntimes_by_quantity(d)): 
            sample = np.random.choice(np.arange(self.get_max_pdf_bin(d) + 1), 
                                      size=(nboots, exp_nsamples[self.closest_exp_time(d, t)]), p = self.dists[d][t,:])
            if sedists is True:
                boot_dists = np.zeros((nboots, self.get_max_pdf_bin(d) + 1))
                for bi in range(nboots):
                    boot_dists[bi,:] = getPdf(sample[bi,:], self.get_max_pdf_bin(d))
                se_dists[t,:] = np.std(boot_dists, 0)
            self.se_quants[d][t] = np.std(np.mean(sample,1),0)
            
        if sedists is True:
            return se_dists
        else:
            return self.se_quants[d]        
    
    
    
    def subdataset(self, seed):
        '''
            Returns an unseeded dataset object, by taking quantities/dists from the s-th seed
        '''         
        
        new_data = self.copy()
        new_data.seeded = False
        new_data.seeds = None
    
        for q in new_data.quantities.keys():
            new_data.quantities[q] = new_data.quantities[q][seed,...]
            
        for d in new_data.dists.keys():
            new_data.dists[d] = new_data.dists[d][seed,...]
            new_data.nsamples[d] = new_data.nsamples[d][seed,...]
            if hasattr(new_data.se_quants, d):
                new_data.se_quants[d] = new_data.se_quants[d][seed,...]
        
        if hasattr(new_data, 'pop_quantities'):
            for q in new_data.pop_quantities.keys():
                new_data.pop_quantities[q] = new_data.pop_quantities[q][seed,...]
            
        if hasattr(new_data, 'pop_dists'):
            for d in new_data.pop_dists.keys():
                new_data.pop_dists[d] = new_data.pop_dists[d][seed,...]
                new_data.pop_nsamples[d] = new_data.pop_nsamples[d][seed,...]
        
        if hasattr(new_data, 'custom_quants'):
            for q in new_data.custom_quants.keys():
                new_data.custom_quants[q] = new_data.custom_quants[q][seed,...]
                new_data.nsamples[q] = new_data.nsamples[q][seed,...]

        return new_data        



    def compute_clonal_quantities(self, livesize = None, subpop = None, full = False):
        ''' 
            Computes clonal quantities from raw_data

            Can extract sub-population by specifying a function name from clones, specifying a function on
                livesize to extract the sub-population, or by specifying an integer corresponding the population index
        '''

        if livesize is None:
            livesize = self.raw_data['liveCS']

        assert(self.collapsed is True)
        assert(self.seeded is False)

        if subpop is not None:
            prefix = str(subpop) + '.'
        else:
            prefix = ''

        if self.seeded:
            print('Not implemented')
            assert(0)
        else:
            if self.type == 'sim':
                clone_anal = clones.sanalysis(livesize = livesize, skip = self.skip, subpop = subpop, 
                                              exp_times = self.exp_times(prefix + 'basalCS'), verbose = False,
                                                **self.compute_kwargs(subpop))
            else:
                clone_anal = clones.xanalysis(livesize = livesize, subpop = subpop, verbose = False,
                                                **self.compute_kwargs(subpop))
                
            if subpop is None:
                self.raw_data['clone_anal'] = clone_anal


            if prefix + 'basalCS' in self.toCalculate:
                self.set_quantity( prefix + 'basalCS', clone_anal.av_bas_cs)
                self.set_se_quants(prefix + 'basalCS', clone_anal.se_bas_cs)
                self.set_nsamples(prefix + 'basalCS', clone_anal.nsamples)
                self.set_dist(prefix + 'basalCS', clone_anal.bas_pdf(self.get_max_pdf_bin(prefix + 'basalCS')))

            if prefix + 'supraCS' in self.toCalculate:
                self.set_quantity( prefix + 'supraCS', clone_anal.av_sup_cs)
                self.set_se_quants(prefix + 'supraCS', clone_anal.se_sup_cs)
                self.set_nsamples(prefix + 'supraCS', clone_anal.nsamples)
                self.set_dist(prefix + 'supraCS', clone_anal.sup_pdf(self.get_max_pdf_bin(prefix + 'supraCS')))

            if prefix + 'totalCS' in self.toCalculate:
                self.set_quantity( prefix + 'totalCS', clone_anal.av_tot_cs)
                self.set_se_quants(prefix + 'totalCS', clone_anal.se_tot_cs)
                self.set_nsamples(prefix + 'totalCS', clone_anal.nsamples)
                self.set_dist(prefix + 'totalCS', clone_anal.tot_pdf(self.get_max_pdf_bin(prefix + 'totalCS')))
                
            if prefix + 'pers' in self.toCalculate:
                self.set_quantity(prefix + 'pers', clone_anal.pers)
                self.set_se_quants(prefix + 'pers', clone_anal.se_pers)
                self.set_nsamples(prefix + 'pers', clone_anal.nsamples)

                    
            if prefix + 'lcd' in self.toCalculate:
                self.set_quantity( prefix + 'lcd', clone_anal.lcd)
                self.set_se_quants(prefix + 'lcd', clone_anal.se_lcd)    
                self.set_nsamples(prefix + 'lcd', clone_anal.nsamples)

                
            if prefix + 'xfrac' in self.toCalculate:
                self.timeless_quant(prefix + 'xfrac')
                if self.type == 'sim':
                    tvals = self.exp_times(prefix + 'basalCS')
                else:
                    tvals = None
                self.set_dist(prefix + 'xfrac', clone_anal.xfrac(tvals, self.get_max_pdf_bin(prefix + 'xfrac')))
                self.set_nsamples(prefix + 'xfrac', clone_anal.nextinct(tvals = tvals))

            if prefix + 'fullbasalCS' in self.toCalculate:
                self.set_quantity( prefix + 'fullbasalCS', clone_anal.av_full_bas_cs)
                self.set_se_quants(prefix + 'fullbasalCS', clone_anal.se_full_bas_cs)
                self.set_nsamples(prefix + 'fullbasalCS', clone_anal.non_missing)
                self.set_dist(prefix + 'fullbasalCS', clone_anal.full_bas_pdf(self.get_max_pdf_bin(prefix + 'fullbasalCS')))

            if prefix + 'fullsupraCS' in self.toCalculate:
                self.set_quantity( prefix + 'fullsupraCS', clone_anal.av_full_sup_cs)
                self.set_se_quants(prefix + 'fullsupraCS', clone_anal.se_full_sup_cs)
                self.set_nsamples(prefix + 'fullsupraCS', clone_anal.non_missing)
                self.set_dist(prefix + 'fullsupraCS', clone_anal.full_sup_pdf(self.get_max_pdf_bin(prefix + 'fullsupraCS')))

            if prefix + 'fulltotalCS' in self.toCalculate:
                self.set_quantity( prefix + 'fulltotalCS', clone_anal.av_full_tot_cs)
                self.set_se_quants(prefix + 'fulltotalCS', clone_anal.se_full_tot_cs)
                self.set_nsamples(prefix + 'fulltotalCS', clone_anal.non_missing)
                self.set_dist(prefix + 'fulltotalCS', clone_anal.full_tot_pdf(self.get_max_pdf_bin(prefix + 'fulltotalCS')))
        
            
            if not self.alltimes or full:
                if prefix + 'jointCS' in self.toCalculate:
                    self.custom_quants[ prefix + 'jointCS'] = clone_anal.joint_bas_sb
                    self.nsamples[ prefix + 'jointCS'] = clone_anal.nsamples   

                if prefix + 'fulljointCS' in self.toCalculate:
                    self.custom_quants[ prefix + 'fulljointCS'] = clone_anal.full_joint_bas_sb
                    self.nsamples[ prefix + 'fulljointCS'] = clone_anal.non_missing

                #all transitions broken down by clone-type
                if prefix + 'tranCS' in self.toCalculate:
                    self.custom_quants[ prefix + 'tranCS'] = clone_anal.allmoves(max_bas = self.get_max_pdf_bin('basalCS'), max_sup =  self.get_max_pdf_bin('supraCS'))
                    self.custom_quants[ prefix + 'btranCS'] = clone_anal.allmoves_bas(_allmoves = self.custom_quants['tranCS'])
                    self.nsamples[ prefix + 'tranCS']  = clone_anal.nsamples
                    self.nsamples[ prefix + 'btranCS'] = clone_anal.nsamples
            
                if prefix + 'btranCS' in self.toCalculate:
                    self.custom_quants[prefix + 'btranCS'] = clone_anal.allmoves_bas(max_bas = self.get_max_pdf_bin('basalCS'))
                    self.nsamples[prefix + 'btranCS'] = clone_anal.nsamples
                        
                #basal transitions broken down by clone-type    
                if prefix + 'cttranCS' in self.toCalculate:
                    if self.compute_kwargs('cttranCS', 'ctypes') is None:
                        print("Need to specify ctypes in compute_kwargs['cttranCS']")
                        assert(0)
                    
                    self.custom_quants[prefix + 'cttranCS']  = clone_anal.allmoves_by_type(max_bas = self.get_max_pdf_bin('basalCS'), 
                                                            max_sup = self.get_max_pdf_bin('supraCS'), ctypes = self.compute_kwargs('cttranCS', 'ctypes'))
                    self.custom_quants[prefix + 'ctbtranCS'] = clone_anal.allmoves_bas_by_type(_allmoves_ctype = self.custom_quants['cttranCS'],
                                                                                    ctypes = self.compute_kwargs('cttranCS', 'ctypes'))
                    self.nsamples[prefix + 'cttranCS']  = clone_anal.nsamples
                    self.nsamples[prefix + 'ctbtranCS'] = clone_anal.nsamples
                    
            
                if prefix + 'ctbtranCS' in self.toCalculate:
                    self.custom_quants[prefix + 'ctbtranCS'] = clone_anal.allmoves_bas(max_bas = self.get_max_pdf_bin('basalCS'))
                    self.nsamples[prefix + 'ctbtranCS'] = clone_anal.nsamples                   


    
    def closest_exp_time(self, quantity, t):
        '''  
            Default behaviour for closest experimental time is to return the same time
        '''
        return t    

    def set_quantity(self, q, values, s = 0):
        '''
            Setting quantity

        '''
                
        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)

        if self.seeded is True:
            assert(np.shape(self.quantities[q])[1] == len(values)) 
            self.quantities[q][s, :] = values
                         
        else:
            self.quantities[q] = values

        return values
    

    def set_se_quants(self, q, values, s = 0):
        '''
            Setting standard error of quantity
            
            We normalise by control if set to do so
        '''

        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)        
                
        if self.seeded:
            assert(np.shape(self.quantities[q])[1] == len(values)) 
            self.se_quants[q][s, :] = values      
                         
        else:
            self.se_quants[q] = values         
            
        return values    
    
    
    def set_dist(self, d, values, s = 0):
        '''
            Setting distribution, handling indices safely depending on whether seeded
        '''

        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)               
        
        if np.ndim(values) == 1: #if time-lines converts to a timed form
            values = values.copy().reshape(1, -1)

        if self.seeded is True:
            if d in self.dists:
                assert(np.shape(self.dists[d])[1] == values.shape[0])
                assert(np.shape(self.dists[d])[2] == values.shape[1])
            self.dists[d][s, ...] = values
        else:
            if self.collapsed is True:
                self.dists[d] = values

        #Update max pdf bin if not specified
        if d not in self._max_pdf_bin:
            self.fix_max_pdf_bin(d)

        return values
    

    def set_nsamples(self, c, values, s = 0):
        ''' 
            Sets number of samples for a given category
        '''

        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)   

        timeless = (c in self.timeless.keys())

        assert(np.sum(values) > 0)

        if not timeless and np.ndim(values) == 0:
            values = np.array([values])

        if self.seeded:
            if not timeless:
                assert(np.shape(self.nsamples[c])[1] == len(values)) 
                self.nsamples[c][s,:] = values    
            else:
                self.nsamples[c][s] = values
        else:
            self.nsamples[c] = values    

    
    def set_pop_nsamples(self, d, values, s = 0):
        '''
            Sets population number of samples for a given distribution
        '''
        
        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0) 

        if self.seeded is True:
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[2] == np.shape(values, 1)) 
            if self.collapsed is True:
                self.pop_nsamples[d][s, :, :] = values
        else:
            assert(np.shape(self.pop_nsamples[d])[0] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 1)) 
            if self.collapsed is True:
                self.pop_nsamples[d] = values
        
    
    def set_pop_quantity(self, q, values, s = 0):
        '''
            Setting population quantity, handling indices safely 
        '''

        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)         
                
        if self.seeded is True:
            assert(np.shape(self.pop_quantities[q])[1:3] == np.shape(values)) 
            self.pop_quantities[q][s, :, :] = values
        else:
            assert(np.shape(self.pop_quantities[q])[0:2] == np.shape(values)) 
            self.pop_quantities[q] = values
  
        return values    
        
        
    def set_pop_dist(self, q, values, s = 0):
        '''
            Setting population distributions, handling indices safely 
        '''
                
        if not self.collapsed:
            print('Dataset not collapsed')                
            assert(0)     

        if self.seeded is True:
            assert(np.shape(self.pop_dists[q])[1] == np.shape(values)) 
            self.pop_dists[q][s, :, :, :] = values
        else:
            assert(np.shape(self.pop_dists[q])[0:2] == np.shape(values))
            self.pop_dists[q] = values
        return values        
    
    def has_clonal_quants(self):
        ''' 
            Returns if we have clonal quantities
        '''
        
        flag = False
        
        for c in self.toCalculate:
            if self.clonal_cat(c):
                return True
        
        return flag
    

    def clonal_cat(self, c):
        '''
            True if is a clonal category
        '''
        if c in clonal_cats:
            return True
        elif '.' in c and c.split('.')[1] in clonal_cats:
            return True   
        else: 
            return False
            
    
class sim_dataset(dataset):
    '''
        Object for interacting with tissue simulations, 
            can have quantities which relate to suprabasal layer only
            
        Inputting negative tscalevals or shedvals re-scales by characteristic scale,
            so we can specify simulation scale times in physical days per characteristic 
            division to obtain times in sim units
        
        Encapsulated with this we have the ability to re-run simulations of the 
            suprabasal layer
            
            
    '''
    def __init__(self, toCalculate, exp_times, tsteps, seeds = None, skip = 10, npops = 1, tscalevals = 1, shedvals = None, 
                            max_pdf_bin = None, alltimes = True, save_results = False, tshift = 0,
                            sim_params = None, norm_by_control = [], ref_scale = 'tau',
                            sbasrefract = 0.0, compute_kwargs = None, analyse_extinct = False):
        #Set-up
        dataset.__init__(self, toCalculate, max_pdf_bin = max_pdf_bin, compute_kwargs = compute_kwargs)
        self.seeds = seeds
        self.nseeds = len(seeds)
        self.type = 'sim'
        self.times = exp_times #experimental times we use
        self.set_default_times()
        
        if sim_params is None:
            sim_params = {}
        self.sim_params  = sim_params
        
        #Set-up control values (must be scalar)
        self.norm_by_control = norm_by_control #specifies which list of quantities to normalise by average ctrl values
        self.ctrl_vals = {}
                
        #Flags: important  for behaviour of how we operate
        self.seeded = True #indicates we are saving data in seeded format, default
        self.collapsed = False #flag indicating whether we have a grid over additional dimensions
        self.alltimes = alltimes #flag to indicate whether we save data at all or just specific times
        self.extinct = False #flag for extinction
        self.analyse_extinct = analyse_extinct
        
        #Setting reference scale:
        if ref_scale not in sim_params:
            self.char_scale = 0
        else:
            self.char_scale = sim_params[ref_scale]
        
        #Simulation time-saving
        if tsteps < 0:
            tsteps = int(np.abs(tsteps*self.char_scale))
        if skip < 0:
            skip = int(np.abs(skip*self.char_scale))
            if skip == 0:
                skip = 1
            
        self.tsteps = tsteps
        self.skip   = skip
        self.tshift = tshift
        
       #Values if we allow variation of shedding parameter and tscales:
        #These are fixed
        if tscalevals is None: 
            tscalevals = [1]
        elif np.ndim(tscalevals) == 0:
            tscalevals = [tscalevals]                
        self.tscalevals = np.array(tscalevals)
        self.manage_tscale(ref_scale)
        
        extradims = (len(self.tscalevals),)
        if shedvals is None or len(shedvals) == 0:
            shedvals = [0]
        self.shedvals   = shedvals
        supradims = (len(self.shedvals),)   
        self.sbasrefract = sbasrefract     
        
        #Make pop quantities:
        self.npops = npops
        self.pop_quantities = {} #quantities at each time-point
        self.pop_dists = {} #probability distributions
        self.pop_nsamples = {} #number of samples at each time-point for each population
        self.se_pop_quants = {} #standard error in population quantities
        
        #Saving results data as well
        self.save_results = save_results
        if self.save_results is True:
            self.results = {}
         
        #Set-up simulation quantities
        if seeds is not None:
            self.init_seeded_quantities(seeds, npops, extradims, supradims)

            

    def set_default_times(self):
        '''
            Sets default times to be time 
        '''
        for q in self.toCalculate:
            if q not in self.times:
                if self.clonal_cat(q):
                    self.times[q] = self.times['basalCS']
                elif '.' in q:
                    self.times[q] = self.times[q.split('.')[1]]
                else:
                    self.times[q] = None
            
            
            
    def manage_tscale(self, ref_scale = 'tau', round = 2):
        '''
            Allows ensuring that there are no invalid t-scale values provided.
            
            If any tscale values are negative, then we re-scale by the reference 
                scale which defaults to the average division time
                
            We also save the transformed indices of tscalevals to the old indices
            
            Note: this is only useful upon pre-computing of data
        '''
        
        if len(self.tscalevals) == 1 and self.tscalevals[0] == 1:
            self.new_to_old_tscidx = np.array([0])
            self.tscalemin = 0
            self.tscalemax = np.inf
            return self.tscalevals

        #Re-scaling negative values by reference scale
        if np.min(self.tscalevals) < 0:
            if ref_scale in self.sim_params:
                self.tscalevals[self.tscalevals < 0] = self.char_scale/self.tscalevals[self.tscalevals < 0]
            else:
                print('No reference scale provided, defaulting to 1')
                
        self.tscalevals = np.abs(self.tscalevals)
        idxs = np.argsort(self.tscalevals)
        self.tscalevals = self.tscalevals[idxs]

        #Ensuring values fall within the valid range
        minInc = self.min_inc()
        T = self.max_time() - self.tshift
        self.tscalemin = self.skip/minInc
        self.tscalemax = self.tsteps/T

        print('τ range: ' + str(self.char_scale/self.tscalemax) + ' to ' + str(self.char_scale/self.tscalemin))

        #handle special case of slightly too big:
        if len(self.tscalevals) == 1 and self.tscalemax < np.inf and int(self.tscalevals[0]) == int(self.tscalemax):
            self.tscalevals = np.array([self.tscalemax])
            print('Warning tscalevals in dangerzone')
        
        #Set conversions to the old-tscales
        idxs = idxs[(self.tscalevals >= self.tscalemin) & (self.tscalevals <= self.tscalemax)]
        self.new_to_old_tscidx = idxs
        
        #Set the new tscalevals
        self.tscalevals = self.tscalevals[(self.tscalevals >= self.tscalemin) & (self.tscalevals <= self.tscalemax)]
        self.tscalevals = np.round(self.tscalevals, round)
        
        assert(len(self.tscalevals) > 0), 'No valid tscale values provided'
        
        return self.tscalevals
        
    
    def set_timeless(self, q, values, s = 0):
        '''
            Setting timeless quantities, handling indices safely depending on whether seeded or suprabasal
        '''
        
        if self.seeded is True:
            assert(np.shape(self.timeless[q])[1] == len(values)) 
            self.timeless[q][s, :] = values
            #Normalise by control
            if q in self.norm_by_control:
                self.timeless[q][s, :] /= self.ctrl_vals[q]
        else:
            assert(np.shape(self.timeless[q])[0] == len(values)) 
            self.timeless[q] = values
            #Normalise by control
            if q in self.norm_by_control:
                self.timeless[q] /= self.ctrl_vals[q]
        
        return values


    def set_quantity(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting quantity, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
            
            We normalise by control if set to do so
        '''
                
        if self.seeded is True:
            assert(np.shape(self.quantities[q])[1] == len(values)) 
            
            if self.collapsed is True:
                self.quantities[q][s, :] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][s, :] /= self.ctrl_vals[q]
            elif q in supra_cats:
                self.quantities[q][s, :, ti, sbsi] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][s, :, ti, sbsi] /= self.ctrl_vals[q]                  
            else:
                self.quantities[q][s, :, ti] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q][s, :, ti] /= self.ctrl_vals[q]                  
                         
        else:
            if self.collapsed is True:
                self.quantities[q] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.quantities[q] /= self.ctrl_vals[q]
            else:
                assert(np.shape(self.quantities[q])[0] == len(values)) 
                if q in supra_cats:
                    self.quantities[q][:, ti, sbsi] = values
                    #Normalise by control
                    if q in self.norm_by_control:
                        self.quantities[q][:, ti, sbsi] /= self.ctrl_vals[q]                
                else:
                    self.quantities[q][:, ti] = values
                    #Normalise by control
                    if q in self.norm_by_control:
                        self.quantities[q][:, ti] /= self.ctrl_vals[q]                
            
        
        return values
    

    def set_se_quants(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting standard error of quantity, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
            
            We normalise by control if set to do so
        '''
                
        if self.seeded is True:
            assert(np.shape(self.quantities[q])[1] == len(values)) 
            
            if self.collapsed is True:
                self.se_quants[q][s, :] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.se_quants[q][s, :] /= self.ctrl_vals[q]
            elif q in supra_cats:
                self.se_quants[q][s, :, ti, sbsi] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.se_quants[q][s, :, ti, sbsi] /= self.ctrl_vals[q]                  
            else:
                self.se_quants[q][s, :, ti] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.se_quants[q][s, :, ti] /= self.ctrl_vals[q]                  
                         
        else:
            if self.collapsed is True:
                self.se_quants[q] = values
                #Normalise by control
                if q in self.norm_by_control:
                    self.se_quants[q] /= self.ctrl_vals[q]
            else:
                assert(np.shape(self.se_quants[q])[0] == len(values)) 
                if q in supra_cats:
                    self.se_quants[q][:, ti, sbsi] = values
                    #Normalise by control
                    if q in self.norm_by_control:
                        self.se_quants[q][:, ti, sbsi] /= self.ctrl_vals[q]                
                else:
                    self.se_quants[q][:, ti] = values
                    #Normalise by control
                    if q in self.norm_by_control:
                        self.se_quants[q][:, ti] /= self.ctrl_vals[q]                
            
        
        return values    
    
    
    def set_dist(self, d, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting distribution, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
        
        if np.ndim(values) == 1:
            values = values.copy().reshape(1, -1)

        if self.seeded is True:
            if d in self.dists:
                assert(np.shape(self.dists[d])[1] == len(values)) 

            if self.collapsed is True:
                self.dists[d][s, :, :] = values
            elif d in supra_cats:
                self.dists[d][s, :, :, ti, sbsi] = values
            else:
                self.dists[d][s, :, :, ti] = values
        else:
            if self.collapsed is True:
                self.dists[d] = values
            else:
                assert(np.shape(self.dists[d])[0] == len(values)) 
                if d in supra_cats:
                    self.dists[d][:, :, ti, sbsi] = values
                else:
                    self.dists[d][:, :, ti] = values
        return values
    

    def set_nsamples(self, c, values, s = 0, ti = 0):
        ''' 
            Sets number of samples for a given category
        '''

        timeless = (c in self.timeless.keys())

        #assert(np.sum(values) > 0)

        if not timeless and np.ndim(values) == 0:
            values = np.array([values])

        if self.seeded is True:
            if not timeless:
                assert(np.shape(self.nsamples[c])[1] == len(values)) 
                if self.collapsed is True:
                    self.nsamples[c][s,:] = values    
                else:
                    self.nsamples[c][s, :, ti] = values
            else:
                self.nsamples[c][s] = values
        else:
            if not timeless:
                if self.collapsed is True:
                    self.nsamples[c] = values    
                else:
                    assert(np.shape(self.nsamples[c])[0] == len(values)) 
                    self.nsamples[c][:, ti] = values
            else:
                self.nsamples[c] = values

    
    def set_pop_nsamples(self, d, values, s = 0, ti = 0):
        '''
            Sets population number of samples for a given distribution
        '''
        if self.seeded is True:
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[2] == np.shape(values, 1)) 
            if self.collapsed is True:
                self.pop_nsamples[d][s, :, :] = values
            else:
                self.pop_nsamples[d][s, :, :, ti] = values
        else:
            assert(np.shape(self.pop_nsamples[d])[0] == np.shape(values, 0)) 
            assert(np.shape(self.pop_nsamples[d])[1] == np.shape(values, 1)) 
            if self.collapsed is True:
                self.pop_nsamples[d] = values
            else:
                self.pop_nsamples[d][:, ti] = values
        
    
    def set_pop_quantity(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting population quantity, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
                
        if self.seeded is True:
            assert(np.shape(self.pop_quantities[q])[1:3] == np.shape(values)) 
            if self.collapsed is True:
                self.pop_quantities[q][s, :, :] = values
            elif q in supra_cats:
                self.pop_quantities[q][s, :, :, ti, sbsi] = values
            else:
                self.pop_quantities[q][s, :, :, ti] = values
        else:
            assert(np.shape(self.pop_quantities[q])[0:2] == np.shape(values)) 
            if self.collapsed is True:
                self.pop_quantities[q] = values
            elif q in supra_cats:
                self.pop_quantities[q][:, :, ti, sbsi] = values
            else:
                self.pop_quantities[q][:, :, ti] = values
        return values    
        
        
    def set_pop_dist(self, q, values, s = 0, ti = 0, sbsi = 0):
        '''
            Setting population distributions, handling indices safely depending on whether seeded or suprabasal
                for a certain value of tscale and suprabasal shed (if required)
        '''
                
        if self.seeded is True:
            assert(np.shape(self.pop_dists[q])[1] == np.shape(values)) 
            if self.collapsed is True:
                self.pop_dists[q][s, :, :, :] = values
            elif q in supra_cats:
                self.pop_dists[q][s, :, :, :, ti, sbsi] = values
            else:
                self.pop_dists[q][s, :, :, :, ti] = values
        else:
            assert(np.shape(self.pop_dists[q])[0:2] == np.shape(values))
            if self.collapsed is True:
                self.pop_dists[q] = values
            elif q in supra_cats:
                self.pop_dists[q][:, :, :, ti, sbsi] = values
            else:
                self.pop_dists[q][:, :, :, ti] = values
        return values   
        
                
    def all_quantities(self):
        return {**self.quantities, **self.dists, #**self.samples, 
                **self.pop_quantities, **self.pop_dists,
                **self.timeless}# , **self.pop_samples}
    
            
    def ntimes_by_quantity(self, quantity):
        '''
            Extracts the numbers of times that are saved
        '''
        
        if self.is_timeless(quantity):
        #if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return 1

        if self.alltimes is True:
            if self.isskipped(quantity) is True:
                ntimes = int(np.ceil((self.tsteps + 1)/self.skip))
            else:
                ntimes = self.tsteps + 1                
        else:
            ntimes = len(self.timeq(quantity))
            
        return ntimes
    
    
    def exp_times(self, quantity, tscale = None, tshift = None):
        '''
            Extract time-stamps of the experimental times
        '''

        if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return None
        
        if tscale is None:
            tscale = self.tscalevals[0]
            
        if tshift is None:
            tshift = self.tshift
        
        if not self.alltimes:
            return np.arange(len(self.timeq(quantity))) 
        else:
            if self.isskipped(quantity) is True:
                return exp2simSkippedTimes(self.timeq(quantity), tscale, -tshift, self.skip)
            else:
                return exp2simTimes(self.timeq(quantity), tscale, -tshift)
            
    
    
    def timeslice_by_quantity(self, quantity, tscale, tshift, alltimes = None):
        '''
            Time-slice for selecting from time-vector of quantity
        '''

        if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return slice(None,None,None)     
        
        if alltimes is None:
            alltimes = self.alltimes        
        
        if alltimes is False:
            if self.isskipped(quantity) is True:
                tslice = exp2simSkippedTimes(self.timeq(quantity), tscale, -tshift, self.skip)
            else:
                tslice = exp2simTimes(self.timeq(quantity), tscale, -tshift)
        else:
            tslice = slice(None,None,None)
            
        return tslice        
    
    
    def tvals_by_quantity(self, quantity, tscale, tshift, alltimes = None):
        '''
            tvals returns the simulation times for alltimes is False
                else returns None to select alltimes
        '''

        if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return None        
        
        if alltimes is None:
            alltimes = self.alltimes
        
        if alltimes is False:
            if self.isskipped(quantity) is True:
                tvals = exp2simSkippedTimes(self.timeq(quantity), tscale, -tshift, self.skip)*self.skip
            else:
                tvals = exp2simTimes(self.timeq(quantity), tscale, -tshift)
        else:
            tvals = None
            
        return tvals    
    
    
    def tvec_by_quantity(self, quantity, tscale = None, tshift = None):
        '''
            Gets the time-vector to plot alongside the time-course data.
        '''
        
        if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return zero_arr

        if np.ndim(tscale) == 0 or len(tscale) == 1:
            tscale = self.tscalevals[0]
        else:
            tscale = 1.0
        
        if tshift is None:
            tshift = self.tshift
        
        if self.alltimes:
            if self.isskipped(quantity) is True:
                skip = self.skip
            else:
                skip = 1
                
            tvals = np.arange(0, self.tsteps + 1, skip)
        else:
            tvals = self.tvals_by_quantity(quantity, tscale, tshift)
        
        return tvals/tscale + tshift
        
    
    def closest_exp_time(self, quantity, t, tscale = None, tshift = None):
        '''
            Obtains the index of the closest experimental time by simulation time
        '''

        if self.timeq(quantity) is None or self.timeq(quantity) is zero_arr: #handling timeless
            return 0           
        
        if tshift is None:
            tshift = self.tshift
            
        if tscale is None:
            tscale = self.tscalevals[0]
            
        idxless = np.where(self.tvec_by_quantity(quantity, tscale, tshift)[t] <= self.timeq(quantity))[0]
        if len(idxless) == 0:
            idx = len(self.timeq(quantity)) - 1
        else:
            idx = idxless[0]
            
        return idx
        
        
    def discrete_pdf(self, d):
        '''
            Returns whether distribution is a discrete distribution
                To store continuous data len(dist_bins) = len(dist) + 1
                To store discrete data len(dist_bins) = len(dist)            
        '''
        if self.dist_bins[d] is None:
            return True
        else:
            if len(self.dist_bins[d]) == self.dists[d].shape[int(self.seeded) + 1]:
                return True
            elif len(self.dist_bins[d]) == self.dists[d].shape[int(self.seeded) + 1] + 1:
                return False
            else:
                assert(0), 'distribution' + d + 'has invalid length'
        
    
            
    def init_seeded_quantities(self, seeds, npops = 1, extradims = (1,), supradims = (1,)):
        '''
            Sets up data structures to compare simulation quantities across seeds
                
            Extra-dimensions allows you to also save data by t-scale value, but also other
                variables you might want to loop over 
            
            Supradims allows you to store
                suprabasal shedding value 
                (for total clonal dynamics)
                
            fixntimes: specifies whether to use experimental times (ntimes = None) 
                or use simulation times (e.g. use ntimes = tsteps) for which we
                can decimate later. This is useful when optimising tscale/shed at
                the end
                
            save_results, which is needed for example if we re-run the suprabasal layer
            
            To use alltimes specify: extradims = (1,), supradims = (1,)
                We have 
            
        '''
        
        #convert to struct
        if np.ndim(extradims) == 0:
            extradims = (extradims,)
            
        #We default to the collapsed form of data
        if self.alltimes is True:
            self.collapsed = True #we default to the collapsed form
            extradims = ()
            supradims = ()
            #self.tscalevals = [1] #we default to the collapsed form
            #self.shedvals = [0]
        
        nseeds = len(seeds)
        
        if self.has_clonal_quants():
            #these store clone-analysis objects, which are build off live-size
            #this data allows efficient computation of clonal statistics 
            #and re-running of the suprabasal layer (only for seeded data)
            self.raw_data['liveCS']     = []
            self.raw_data['clone_anal'] = []

            #Note clonal_quants are now all caclulated later so we don't pre-store
    
        if 'density' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('density')
            self.quantities['density'] = np.zeros((nseeds, ntimes, *extradims))
            self.pop_quantities['density'] = np.zeros((nseeds, npops, ntimes, *extradims)) 

        if 'divrate' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('divrate')
            self.quantities['divrate'] = np.zeros((nseeds, ntimes, *extradims))

        if 'death' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('death')
            self.quantities['death'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'T1s' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('T1s')
            self.quantities['T1s'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'MSD' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('MSD')
            self.quantities['MSD'] = np.zeros((nseeds, ntimes, *extradims))
            
        if 'shape' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('shape')
            self.quantities['shape'] = np.zeros((nseeds, ntimes, *extradims))

        if 'area' in self.toCalculate:
            ntimes = self.ntimes_by_quantity('area')
            self.quantities['area'] = np.zeros((nseeds, ntimes, *extradims))


        if 'fate_reff' in self.toCalculate:
            self.timeless['fate_reff'] = np.zeros((nseeds, 3, *extradims))
            self.nsamples['fate_reff'] = np.zeros((nseeds), dtype = int)


    def ntscales_to_save(self, tscalevals = None):
        '''
            Returns the number of t-scales to save
            Depends on whether we save collapsed or not            
        '''
        if tscalevals is None:
            tscalevals = self.tscalevals
            
        if self.collapsed or self.alltimes:
            return 1
        else:
            return len(tscalevals)


    def nsheds_to_save(self, shedvals):
        '''
            Returns the number of shedding values to save
            Depends on whether we save collapsed or not
        '''
        if shedvals is None:
            shedvals = self.shedvals
        
        if self.collapsed or self.alltimes:
            return 1
        else:
            return len(shedvals)
    
        
    def quantities_by_seed(self, tiss, results, seed):
        '''

        For a given simulation, seed, and array of self.tscalevals
        Uses experimental time values for specified self.tscalevals and tshift
        And adds them to data structures by seed

        Indicates whether the re-running of the suprabasal layer should occur here

        Can store non-decimated data for optimization of self.tscalevals.
        '''
        
        #Add to tracked seeds
        self.added_seeds.add(seed)

        if self.save_results:
            self.results[seed] = results
            
        #Only consider simulations which haven't reach extinction     
        if np.sum(results['ncells'][-1,:]) <= 5:
            print('Simulation went extinct, cannot analyse')
            self.extinct = True
            if self.analyse_extinct is False:
                return self
        elif np.sum(results['ncells'][-1,-1]) == 0:
            print('Stem population went extinct, cannot analyse')
            self.extinct = True
            if self.analyse_extinct is False:
                return self
        
        #Check skip is correct
        if 'skip' in results.keys() and results['skip'] != self.skip:
            print('Warning: skip value in results does not match dataset skip value')
            #assert(0)
            self.skip = results['skip']
        
        #Set characteristic scale
        if self.char_scale <= 0:
            self.char_scale = get_division_scale(tiss)
            print('Using reference scale:', self.char_scale)
        
        #Set-up control values
        for q in self.norm_by_control:
            assert('control' in results.keys())
            assert(q in results['control'].keys())            
            self.ctrl_vals[q] = results['control'][q]

   
        #Calculate clonal dynamics
        if self.has_clonal_quants():
            clone_anal = clones.sanalysis(results, skip = self.skip) #analysis
            self.raw_data['liveCS'].append(clone_anal.livesize)
            self.raw_data['clone_anal'].append(clone_anal)
            

        if 'density' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tslice = self.timeslice_by_quantity('density', self.tscalevals[ti], self.tshift)
                self.set_quantity('density', np.sum(results['ncells'], 1)[tslice], seed, ti = ti)
                self.set_pop_quantity('density', results['ncells'][tslice, :].transpose(), seed, ti = ti)
     
                
        if 'divrate' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('divrate', self.tscalevals[ti], self.tshift)
                divrate_values = simDivrate(tiss, results, times=tvals)
                self.set_quantity('divrate', divrate_values, seed, ti=ti)
                
        if 'death' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('death', self.tscalevals[ti], self.tshift)
                death_values = simDeaths(tiss, times=tvals)
                self.set_quantity('death', death_values, seed, ti=ti)
                
        #'T1s', 'MSD', 'shape'
        if 'T1s' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('T1s', self.tscalevals[ti], self.tshift)
                T1_values = sa.num_T1s(results['shapes']['jlengths'], results['cellid'])
                self.set_quantity('T1s', T1_values, seed, ti=ti)
                
        if 'MSD' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('MSD', self.tscalevals[ti], self.tshift)
                MSD_values = sa.msddiv(results['ncells'], results['anc'], results['cellid'], results['cellpos'], self.skip)
                self.set_quantity('MSD', MSD_values, seed, ti = ti)
                
        if 'shape' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('shape', self.tscalevals[ti], self.tshift)
                shape_values = sa.p0(results['shapes']['areas'], perims = results['shapes']['perims'])
                self.set_quantity('shape', np.array([np.mean(p0s) for p0s in shape_values]), seed, ti = ti)

        if 'area' in self.toCalculate:
            for ti in range(self.ntscales_to_save()):
                tvals = self.tvals_by_quantity('area', self.tscalevals[ti], self.tshift)
                self.set_quantity('area', np.array([np.mean(np.abs(As)) for As in results['shapes']['areas']]), seed, ti = ti)


        if 'fate_reff' in self.toCalculate:
            fracs, samples = analysis.model_reff(results, returnFull = True)
            self.set_timeless('fate_reff', fracs, seed)
            self.set_nsamples('fate_reff', samples, seed)

        return self



    def rerun_suprabasal(self, sb = None, sbasrefract = 0, seed = None, sbsi = 0, 
                         results = None, tiss = None, tscalevals = None, clone_anal = None, 
                         shed_profile = None, sbas_cv = 1.0):
        '''
            Runs the suprabasal layer:
                Can either run the seeded version (using slow method)
                Or the unseed
            
            
            Runs suprabasal layer once and save's the necessary suprabasal quantities for an
                array or individual tscales
                
            Note this will overwrite data in these indices.
            
            shed_profile: time-varying shed rate
            
            
        '''
        
        if sb is None:
            shedvals = self.shedvals
        else:
            shedvals = [sb]
            
        #We don't rerun suprabasal layer, if we have no shedding
        if shedvals[0] == 0 and sbasrefract == 0 and len(shedvals) <= 1:
            return       
        
        if self.seeded is True:           
            if seed is not None:
                seeds = [seed]
            else:
                seeds = self.seeds
                
            for seed in seeds:
                for sb in shedvals:
                    self.rerun_suprabasal_seeded(results = results, tiss = tiss, sb = sb, seed = seed, sbsi = sbsi, 
                         tscalevals = tscalevals, clone_anal = clone_anal, shed_profile = shed_profile, 
                         sbasrefract = sbasrefract, sbas_cv = sbas_cv)
        else:
            print('Not implemented')
            assert(0)
        
        return self
        

        
    def rerun_suprabasal_seeded(self, results = None, tiss = None, sb = 0, seed = 0, sbsi = 0, 
                         tscalevals = None, clone_anal = None, shed_profile = None, sbasrefract = 0,
                         sbas_cv = 1.0):
        '''
            Runs suprabasal layer once and save's the necessary suprabasal quantities for an
                array or individual tscales
                
            Note this will overwrite data in these indices.
            
            shed_profile: time-varying shed rate
            
            
        '''
        
        assert(self.seeded is True)
        
        if clone_anal is None:
            if results is not None:
                clone_anal = clones.sanalysis(results, skip = self.skip) 
            else:
                if len(self.raw_data['clone_anal']) < seed:
                    print('Too few seeds provided for clonal analysis')
                    assert(0)
                clone_anal = self.raw_data['clone_anal'][seed] #extra data for clonal analysis
        
        #Set tscalevals
        if tscalevals is None:
            tscalevals = self.tscalevals
        else:
            assert(len(tscalevals) <= len(self.tscalevals))
        
        if sb < 0:
            sb *= -self.char_scale
            
        #can use negative fractional shed rate
        if sbasrefract < 0:
            if sb <= 0:
                sbasrefract = 0
            else:
                refract = -sbasrefract
                sbasrefract = (sb - 1)*refract
                sb = (sb - 1)*(1 - refract) + 1
                assert(sb > 0)
                assert(sbasrefract > 0)
            
             
        np.random.seed(supraseed(seed, self.char_scale))

        if shed_profile is None:
            clone_anal.rerun_suprabasal(sb, sbasrefract, sbas_cv = sbas_cv)
        else: #Old method, full re-running of suprabasal layer
            results = tiss.rerunsuprabasal(results, tiss.time, sb, skip = self.skip, shed_profile = shed_profile, 
                                        refract = sbasrefract)
            results['tiss'] = tiss
            
            if self.save_results:
                self.results[seed] = results
            
            clone_anal.recalc_supra_livesize(results)
        
        self.raw_data['liveCS'][seed]     = clone_anal.livesize #update live-size to the current one   
        self.raw_data['clone_anal'][seed] = clone_anal

        return results
    
    

    
    
    def calculate_combined_quantities(self, use_wam = True, full = False):
        '''
            Returns an simulation dataset object.
            Takes averages of samples for pdfs and quantities
            
            Takes data from different simulation seeds and combines them into calculated
                quantities that we want from simulations
            
            We use a weighted average unless use_wam = False

            full = True, computes even custom quantities
        '''
        
        sim_data = self.copy()
        sim_data.seeded = False
        sim_data.seeds = None
    
        for q in sim_data.quantities.keys():
            sim_data.combine_quantity(q, use_wam)
            
        for d in sim_data.dists.keys():
            sim_data.combine_dist(d, use_wam)
                
        for q in sim_data.pop_quantities.keys():
            if use_wam is True and q in sim_data.nsamples and np.sum(sim_data.nsamples[q]) > 0:
                sim_data.pop_quantities[q] = np.average(sim_data.pop_quantities[q], 0, weights = sim_data.pop_nsamples[q])
            else:
                sim_data.pop_quantities[q] = np.mean(sim_data.pop_quantities[q],0)
            
        for d in sim_data.pop_dists.keys():
            if use_wam is True and np.sum(sim_data.nsamples[d]) > 0:
                sim_data.pop_dists[d] = np.average(sim_data.pop_dists[d], 0, weights = self.pop_nsamples[d])
            else:
                sim_data.pop_dists[d] = np.mean(sim_data.pop_dists[d],0)  
            sim_data.pop_nsamples[d] = np.sum(self.pop_nsamples[d], 0)
            
        #calculate custom quantities from raw data
        sim_data.unseed_custom_quantities()

        #Compute the rest of the quantitites
        sim_data.calculate_custom_quantities(full = full)
            
        return sim_data
    
    
    def combine_timeless(self, q, use_wam = True):
        '''
            Combines timeless quantities
        '''

        self.se_quants[q] = np.std(self.timeless[q],0)

        if use_wam is True and q in self.nsamples and np.sum(self.nsamples[q]) > 0:
            self.quantities[q] = np.average(self.timeless[q], 0, weights = self.nsamples[q])
        else:
            self.timeless[q] = np.mean(self.timeless[q],0)




    def combine_quantity(self, q, use_wam = True):
        ''' 
            Combines quantities from different seeds
        '''
       
        #Calculate SE in simplest way
        if q not in self.dists: #there is no weighting of std by samples
            self.se_quants[q] = np.std(self.quantities[q],0) 
            
        #Calculate quantity using weighted or unweighted average           
        if use_wam is True and q in self.nsamples and np.sum(self.nsamples[q]) > 0:
            if q in supra_cats and self.collapsed is False:
                #Makes sure dimensions of weights are correct
                weights = np.repeat(self.nsamples[q].reshape(*np.shape(self.nsamples[q]),1), 
                                    self.quantities[q].shape[-1], axis = -1)
            else:
                weights = self.nsamples[q]

            assert(np.sum(weights) > 0), q + ' has no samples' +  str(self.quantities[q]) +  str(weights)
            self.quantities[q] = np.average(self.quantities[q], 0, weights = weights)
        else:
            self.quantities[q] = np.mean(self.quantities[q],0)
    
    
    def combine_dist(self, d, use_wam = True):
        ''' 
            Combines distributions from different seeds
        '''    
        if use_wam is True and np.sum(self.nsamples[d]) > 0:
            if d in supra_cats and self.collapsed is False:
                #Makes sure dimensions of weights are correct   
                weights = np.repeat(self.nsamples[d].reshape(*self.nsamples[d].shape[:-1],1,self.nsamples[d].shape[-1]), 
                                    self.dists[d].shape[-3], axis = -2)                                     
                weights = np.repeat(weights.reshape(*np.shape(weights),1), 
                                    self.dists[d].shape[-1], axis = -1)
            elif not self.collapsed:
                #Add dimension along size axis                    
                weights = np.repeat(self.nsamples[d].reshape(*self.nsamples[d].shape[:-1],1,self.nsamples[d].shape[-1]), 
                                    self.dists[d].shape[-2], axis = -2)
            else: #collapsed is True
                weights = np.repeat(np.expand_dims(self.nsamples[d],axis=-1), self.dists[d].shape[-1], axis=-1)
                
            self.dists[d] = np.average(self.dists[d], 0, weights = weights)
        else:
            self.dists[d] = np.mean(self.dists[d],0)     
    

        #Calculate number of samples and updated SD quants
        self.nsamples[d] = np.sum(self.nsamples[d], 0)
        self.se_quants[d] = self.sd_dist(d, True)
        
    
    def unseed_custom_quantities(self):
        '''
            Calculates unseeded form for data for custom quantities
            
            We need to specify how to do this for each type of raw data        
        '''
        
        if 'liveCS' in self.raw_data: 
            #Combine across simulations
            self.raw_data['liveCS'] = np.vstack([d for d in self.raw_data['liveCS']])
            self.raw_data.pop('clone_anal') #We remove as we don't use for further calculations
    
    
    
    
    def calculate_custom_quantities(self, tscale = None, full = False):
        ''' 
            Calculates custom quantities, data should already be unseeded
        
            clonal_kwargs: additional arguments for clonal analysis

            Computes for each sub-population
        '''

        if tscale is None:
            tscale = self.tscalevals[0]
        
        if self.seeded is True:
            print('Warning: custom quantities only work for unseeded data')
            return
        
        assert(self.seeded is False)
        assert(self.collapsed is True)
            

        if self.has_clonal_quants():
            tslice = self.timeslice_by_quantity('basalCS', tscale, self.tshift)
            livesize = self.raw_data['liveCS'][:,tslice,:]
            for subpop in self.subpops:
                #calculate cloness
                self.compute_clonal_quantities(livesize, subpop, full = full)
    
    
    def sd_dist(self, d, useSE = False):
        '''
            Calculate the standard deviation of a probability distribution
                Assumes that the average quantity is the mean and is up to date
                Also that nsamples is up to date if we using useSE = True
                
            If useSE is True, then we use the standard error of the mean
        '''
        if self.dist_bins[d] is None:
            bins = np.arange(0, self.get_max_pdf_bin(d) + 1)
        elif self.discrete_pdf(d) is True:
            bins = self.dist_bins[d]
        else:
            #use-bin centres for numerical integration of continuous pdf
            bins = (self.dist_bins[d][:-1] + self.dist_bins[1:])/2
            
        #calculate the standard deviation, making the dimensions work
        if self.seeded:
            if self.collapsed is True:
               sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:] - self.quantities[d][:,:,None])**2, axis = int(self.seeded) + 1))               
            else:
                if d in supra_cats:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:,None,None] - self.quantities[d][:,:,None,:,:])**2, axis = int(self.seeded) + 1))               
                else:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,None,:,None] - self.quantities[d][:,:,None,:])**2, axis = int(self.seeded) + 1))                                                   
        else:
            if self.collapsed is True:
               sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:] - self.quantities[d][:,None])**2, axis = int(self.seeded) + 1))               
            else:
                if d in supra_cats:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:,None,None] - self.quantities[d][:,None,:,:])**2, axis = int(self.seeded) + 1))               
                else:
                   sd = np.sqrt(np.sum(self.dists[d]*(bins[None,:,None] - self.quantities[d][:,None,:])**2, axis = int(self.seeded) + 1))                                                   

        
        #convert to the standard error of the mean
        if useSE:
            if d in supra_cats and self.collapsed is False:
                sd /= np.sqrt(self.nsamples[d][...,None])
            else:
                sd /= np.sqrt(self.nsamples[d])

        return sd
    
    
    def data_time_crop(self, tscale = None, ti = 0, verbose = True):
        '''
            Takes each simulated data in self.toCalculate and crops time course to the appropriate 
                length and returns a copy
                            
        '''
        
        if tscale is None:
            tscale = self.tscalevals[ti]
            
            
        if tscale < 0:
            tscale = -self.char_scale/tscale
            
        if tscale < self.tscalemin or tscale > self.tscalemax:
            print('tscale out of range, returning None (silence with verbose = False)')
            return None
        
        sim_data = self.copy()
            
        sim_data.tscalevals = [tscale]
        
        if sim_data.alltimes is False:
            print('Already cropped')
            return sim_data
        
        sim_data.alltimes = False

        for qtype in ['quantities', 'dists', 'nsamples', 'se_quants']:
            for quant in getattr(sim_data, qtype):
                tslice = sim_data.timeslice_by_quantity(quant, tscale, sim_data.tshift, alltimes = False)
                if sim_data.seeded:
                    getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, tslice, ...]
                else:
                    getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][tslice, ...]

        for qtype in ['pop_quantities', 'pop_dists', 'pop_nsamples']:
            for quant in getattr(sim_data, qtype):
                tslice = sim_data.timeslice_by_quantity(quant, tscale, sim_data.tshift)
                if sim_data.seeded:
                    getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, :, tslice, ...]
                else:
                    getattr(sim_data, qtype)[quant] = getattr(sim_data, qtype)[quant][:, tslice, ...]

        #Compute custom quantities for full and for each specified subpopulation
        sim_data.calculate_custom_quantities(tscale)

        return sim_data
    
        
    
    def get_instance(self, ti = 0, sbsi = 0):
        '''
            Get's a particular dataset for certain tscale-values and suprabasal shed values
        '''
        
        assert(self.alltimes is False)
        if self.collapsed is True:
            print('Already collapsed')
            return self

        if self.seeded is True:
            sim_data = self.get_seeded_instance(ti, sbsi)
        else:
            sim_data = self.copy()
            sim_data.collapsed = True  
                        
            for q in sim_data.quantities.keys():
                if q in supra_cats:
                    sim_data.quantities[q] = sim_data.quantities[q][:,ti,sbsi]
                    if hasattr(sim_data.se_quants, q):
                        sim_data.se_quants[q]  = sim_data.se_quants[q][:,ti,sbsi]
                else:
                    sim_data.quantities[q] = sim_data.quantities[q][:,ti]
                    if hasattr(sim_data.se_quants, q):                    
                        sim_data.se_quants[q]  = sim_data.se_quants[q][:,ti]
                    
            for d in sim_data.dists.keys():
                if d in supra_cats:
                    sim_data.dists[d] = sim_data.dists[d][:,:,ti,sbsi]
                else:
                    sim_data.dists[d] = sim_data.dists[d][:,:,ti]  
                                        
            for q in sim_data.pop_quantities.keys():
                if q in supra_cats:
                    sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,ti,sbsi]
                else:
                    sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,ti]
                                
                
            for d in sim_data.pop_dists.keys():
                if d in supra_cats:
                    sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,ti,sbsi]
                else:
                    sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,ti]                
    
    
            for d in sim_data.dists.keys():                    
                sim_data.nsamples[d] = sim_data.nsamples[d][:,ti]
    
    
        return sim_data
    
    
    def get_seeded_instance(self, ti = 0, sbsi = 0):
        '''
            Get's a particular dataset for certain tscale-values and suprabasal shed values
                when data is in the seeded formal
        '''
        
        sim_data = self.copy()
        sim_data.collapsed = True 
        
        sim_data.tscalevals = [self.tscalevals[ti]]
        sim_data.shedvals = [self.shedvals[sbsi]] 

        for q in sim_data.quantities.keys():
            if q in supra_cats:
                sim_data.quantities[q] = sim_data.quantities[q][:,:,ti,sbsi]
                if hasattr(sim_data.se_quants, q):
                    sim_data.se_quants[q]  = sim_data.se_quants[q][:, :,ti,sbsi]                
            else:
                sim_data.quantities[q] = sim_data.quantities[q][:,:,ti]
                if hasattr(sim_data.se_quants, q):                    
                    sim_data.se_quants[q]  = sim_data.se_quants[q][:,:,ti]                
                
        for d in sim_data.dists.keys():
            if d in supra_cats:
                sim_data.dists[d] = sim_data.dists[d][:,:,:,ti,sbsi]
            else:
                sim_data.dists[d] = sim_data.dists[d][:,:,:,ti]  
            sim_data.nsamples[d] = sim_data.nsamples[d][:,:,ti]
                
        for q in sim_data.pop_quantities.keys():
            if q in supra_cats:
                sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,:,ti,sbsi]
            else:
                sim_data.pop_quantities[q] = sim_data.pop_quantities[q][:,:,:,ti]
                            
            
        for d in sim_data.pop_dists.keys():
            if d in supra_cats:
                sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,:,ti,sbsi]
            else:
                sim_data.pop_dists[d] = sim_data.pop_dists[d][:,:,:,:,ti]                
        
        
        return sim_data
    
    
    def get_tscale_idx(self, tscale):
        if tscale in self.tscalevals:
            return np.where(tscale == self.tscalevals)[0][0]
        else:
            return None        

    def get_shed_idx(self, shed):
        if shed in self.shedvals:
            return np.where(shed == self.shedvals)[0][0]
        else:
            return None        
        
        
    def sample(self, d, t = 0, seed = None, ti = 0, si = 0):
        '''
            Returns samples for a given distribution at a given time-point
                indexed by t for tscale-idx ti and suprabasal shed idx si

            Defaults to: returning combining data across all seeds
            Alternatively can specify a seed
            
            Does not work with data with extra-dimensions
            
        '''
        
        #for collapsed data
        if self.collapsed is True:
            return dataset.sample(self, d, t, seed, ti, si)
          
        #for data with extra dimensions
        if self.seeded is True:
            if seed is None: #use combined over sims
                if d in supra_cats:
                    N = np.sum(self.nsamples[d][:,t, ti, si], 0)
                    hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t, ti, si], axis=0) * N).astype(int)
                else:
                    N = np.sum(self.nsamples[d][:, t, ti], 0)
                    hist_counts = (np.average(self.dists[d][:, t, :], weights=self.nsamples[d][:, t, ti], axis=0) * N).astype(int)
            else:
                if d in supra_cats:
                    N = self.nsamples[d][seed,t, ti, si]
                    hist_counts = (self.dists[d][seed, t, :, ti, si]*N).astype(int)  
                else:
                    N = self.nsamples[d][seed,t, ti]
                    hist_counts = (self.dists[d][seed, t, :, ti]*N).astype(int)
        else:
            if d in supra_cats:
                N = self.nsamples[d][t, ti, si]
                hist_counts = (self.dists[d][t, :, ti, si]*N).astype(int)
            else:
                N = self.nsamples[d][t, ti]
                hist_counts = (self.dists[d][t, :, ti]*N).astype(int)
            
        return np.repeat(np.arange(len(hist_counts)), hist_counts)

    def isskipped(self, q):
        '''
            Determines whether the simulated quantity is skipped
        '''

        if self.clonal_cat(q):
            return True
        elif '.' in q:
            return skipped[q.split('.')[1]]
        else:
            return skipped[q]  
            
        
class exp_dataset(dataset):
    '''
        Object for manipulating experimental data
    '''
    
    def __init__(self, filename, prefix = '', suffix = '', toCalculate = [], max_pdf_bin = None, 
                compute_kwargs = None, max_time = np.inf):  
        dataset.__init__(self, toCalculate, max_pdf_bin = max_pdf_bin, 
                         compute_kwargs = compute_kwargs)        
        data_full = self.load_data_file(filename, prefix, suffix)
        self.extract_quantities(data_full, prefix, suffix, max_time)
        self.fix_max_pdf_bin() #fixes pdf bins based on current
        self.chop_data(max_time)

        self.type = 'exp'        
        self.exp_prefix = prefix
        self.exp_suffix = suffix 
    

    def data_time_crop(self):
        return self

    def chop_data(self, max_time):
        ''' 
            Chop data where the time-point is greater than max_time
        '''

        for q in self.quantities.keys():
            tslice = np.where(self.timeq(q) <= max_time)[0]
            self.quantities[q] = self.quantities[q][tslice]
            self.se_quants[q] = self.se_quants[q][tslice]

        for d in self.dists.keys():
            tslice = np.where(self.timeq(d) <= max_time)[0]
            self.dists[d] = self.dists[d][tslice]

        for q in self.nsamples.keys():
            tslice = np.where(self.timeq(q) <= max_time)[0]
            self.nsamples[q] = self.nsamples[q][tslice]

        for q in self.custom_quants.keys():
            tslice = np.where(self.timeq(q) <= max_time)[0]
            self.custom_quants[q] = self.custom_quants[q][tslice]

        if 'liveCS' in self.raw_data:
            tslice = np.where(self.timeq('basalCS') <= max_time)[0]
            self.raw_data['liveCS'] = self.raw_data['liveCS'][:,tslice,:]

        for q in self.times.keys():
            if self.times[q] is None:
                continue
            tslice = np.where(self.times[q] <= max_time)[0]
            self.times[q] = self.times[q][tslice]

        return
    
    def extract_quantities(self, data_full, prefix = '', suffix = '', max_time = np.inf):
        '''
            Extracts relevant quantities for analyis
                and saves alongside their resulting time-values (normally in days)
        '''
        
        if 'basalCS' in self.toCalculate:
            self.quantities['basalCS'] = np.array(data_full[prefix + 'Basalsize' + suffix])
            self.dists['basalCS'] = np.array(data_full[prefix + 'pdf'][:,0,:])
            #self._max_pdf_bin['basalCS'] = np.shape(self.dists['basalCS'])[1]  #set automatically
            self.dist_bins['basalCS'] = None
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['basalCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullBasalsize' + suffix]
                self.nsamples['basalCS'] = np.array([len(s) for s in samples])
                #self.samples['basalCS'] = data_full[prefix + 'FullBasalsize' + suffix]
            self.times['basalCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['basalCS'] = np.array(data_full[prefix + 'SDBasalsize' + suffix])


        if 'totalCS' in self.toCalculate:
            self.quantities['totalCS'] = np.array(data_full[prefix + 'Totalsize' + suffix])
            self.dists['totalCS'] = np.array(data_full[prefix + 'pdf'][:,2,:])
            #self._max_pdf_bin['totalCS'] = np.shape(self.dists['totalCS'])[1]   
            self.dist_bins['totalCS'] = None     
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['totalCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullBasalsize' + suffix]
                self.nsamples['totalCS'] = np.array([len(s) for s in samples])
                #self.samples['totalCS'] = data_full[prefix + 'FullTotalsize' + suffix]                      
            self.times['totalCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['totalCS'] = np.array(data_full[prefix + 'SDTotalsize' + suffix])

        if 'supraCS' in self.toCalculate:
            self.quantities['supraCS'] = np.array(data_full[prefix + 'Suprasize' + suffix])
            self.dists['supraCS'] = np.array(data_full[prefix + 'pdf'][:,1,:])
            #self._max_pdf_bin['supraCS'] = np.shape(self.dists['supraCS'])[1] 
            self.dist_bins['supraCS'] = None    
            if (prefix + 'Nsamples' + suffix) in data_full.keys():
                self.nsamples['supraCS'] = np.array(data_full[prefix + 'Nsamples' + suffix])
            else:
                samples = data_full[prefix + 'FullSuprasize' + suffix]
                self.nsamples['supraCS'] = np.array([len(s) for s in samples])                     
                #self.samples['supraCS'] = data_full[prefix + 'FullSuprasize' + suffix]
            
            self.times['supraCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['supraCS'] = np.array(data_full[prefix + 'SDSuprasize' + suffix])
            
            
        if 'jointCS' in self.toCalculate:
            if prefix + 'jointpdf' in data_full:
                self.custom_quants['jointCS'] = data_full[prefix + 'jointpdf']
            elif 'livesize' in data_full:
                lcs = clones.lanalysis(data_full['livesize'])
                self.custom_quants['jointCS'] = lcs.joint_pdf
            else:
                print('Data not saved')
                assert(0)
            self.times['jointCS'] = np.array(data_full[prefix + 'TimeClonesize'])

        if 'density' in self.toCalculate:
            self.quantities['density'] = np.array(data_full[prefix + 'Density'])
            self.se_quants['density']  = np.array(data_full[prefix + 'SDDensity' + suffix])
            self.times['density'] = np.array(data_full[prefix + 'TimeDensity'])

        if 'divrate' in self.toCalculate:
            self.quantities['divrate'] = np.array(data_full[prefix + 'Divrate'])/np.mean(data_full['ctrDivrate'])
            self.se_quants['divrate']  = np.array(data_full[prefix + 'SDDivrate' + suffix])
            self.times['divrate'] = np.array(data_full[prefix + 'TimeDivrate'])

        if 'pers' in self.toCalculate:
            if prefix + 'Persistence' in data_full:
                self.quantities['pers'] = np.array(data_full[prefix + 'Persistence'])
                self.times['pers'] = np.array(data_full[prefix + 'TimePers'])
                self.se_quants['pers'] = np.array(data_full[prefix + 'SDPersistence' + suffix])

        if 'lcd' in self.toCalculate:
            self.quantities['lcd'] = np.array(data_full['lcd'])
            self.times['lcd'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.se_quants['lcd'] = np.array(data_full['SDlcd'])


        if 'death' in self.toCalculate:
            self.quantities['death'] = np.array(data_full[prefix + 'Cas3'])
            self.times['death'] = np.array(data_full['timeCas3'])
           
        if 'livesize' in data_full:
            self.times['basalCS'] = np.array(data_full[prefix + 'TimeClonesize'])
            self.raw_data['liveCS'] = data_full['livesize']
            if max_time < self.times['basalCS'][-1]:
                print('Chopping data with t >', max_time)
                self.raw_data['liveCS'][:, self.times['basalCS'] < max_time,:]
            #Compute the rest of the quantities
            for subpop in self.subpops:
                #calculate cloness
                if self.has_clonal_quants():
                    self.compute_clonal_quantities(self.raw_data['liveCS'], subpop)
        

        return
          
                     
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename, deleteEOF = False):
    try:
        with open(filename, 'rb') as file:  # Overwrites any existing file.
            return pickle.load(file)
    except EOFError:
        if deleteEOF:
            print(f"Deleting file due to EOFError: {filename}")
            os.remove(filename)
    except AttributeError as e:
        print(f"Attribute error while processing file {filename}: {e}")
        print('Trying to proceed')
        with open(filename, 'rb') as file:  # Overwrites any existing file.
            return pickle.load(file)        
    except Exception as e:
        print(f"Error while processing file {filename}: {e}")
        return None
    
    
def exp2simTimes(texp, tscale, tshift = 0):
    if np.ndim(texp) == 0:
        return int((texp + tshift)*tscale)
    else:
        return ((texp + tshift)*tscale).astype('int32')


def exp2simSkippedTimes(texp, tscale, tshift = 0, skip = 10):
    if np.ndim(texp) == 0:
        return int((texp + tshift)*tscale/skip)
    else:
        return ((texp + tshift)*tscale/skip).astype('int32')


def join_datasets(dss):
    '''
        Takes a list of datasets from different seeds and then joins
            them into one dataset object
    '''
    data = dss[0]
    
    if len(dss) > 1:   
        for d in dss[1:]:
            if data.added_seeds.isdisjoint(d.added_seeds):
                data = data.add_seeded_dataset(d)
            else:
                print('Joining data with repeated seeds')
                assert(0)
            
    return data


def simDivrate(tiss, results, tsteps = None, times = None, dx = 50):
    '''
        Calculates simulation division rate
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        This is shifted by dx = 50
    '''

    if tsteps is None:
        tsteps = tiss.time

    if times is None:
        times = np.arange(tsteps + 1) 
        
    ndivs = np.zeros(tsteps + 1)
    ndies = np.zeros(tsteps + 1)
    nkils = np.zeros(tsteps + 1)
    ndifs = np.zeros(tsteps + 1)

    for t in range(tsteps):
        for p in range(tiss.npops):
            if t in list(tiss.cellpops[p].dupTime.keys()):
                #print(len(tiss.cellpops[p].dupTime[t]))
                ndivs[t] += len(tiss.cellpops[p].dupTime[t])
                ndifs[t] += np.sum([len(ev) == 2 for ev in tiss.cellpops[p].dupTime[t]])

            if t in list(tiss.cellpops[p].dieTime.keys()):
                #print((tiss.cellpops[p].dieTime[t]))
                ndies[t] += len(tiss.cellpops[p].dieTime[t])

            if t in list(tiss.cellpops[p].kilTime.keys()):
                #print((tiss.cellpops[p].kilTime[t]))
                nkils[t] += len(tiss.cellpops[p].kilTime[t])

    nevnt = ndivs + ndies - nkils - ndifs
    times = times.astype('int')
    return np.convolve(nevnt/np.sum(results['ncells'],1),np.ones(dx)/dx, mode='same')[times]


def simDeaths(tiss, tsteps = None, times = None, dx = 100):
    '''
        Calculates simulation deaths
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        This is shifted by dx = 50
    '''

    if tsteps is None:
        tsteps = tiss.time + 1

    if times is None:
        times = np.arange(tsteps + 1) 
        
    nkils = np.zeros(tsteps + 1)

    for t in range(tsteps):
        for p in range(tiss.npops):
            if t in list(tiss.cellpops[p].kilTime.keys()):
                #print((tiss.cellpops[p].kilTime[t]))
                nkils[t] += len(tiss.cellpops[p].kilTime[t])

    times = times.astype('int')
    return np.convolve(nkils,np.ones(dx)/dx, mode='same')[times]


def get_division_scale(tiss, popidx = None):
    '''
        Get the governing time-scale of division for a multiple population tissue,
            takes the value from the highest indexed population that has a non-zero 
            rate unless we specify population
            
    '''
    time = 0
    if popidx is None:
        for p in reversed(range(tiss.npops)):
            if tiss.cellpops[p].divtime > 0:
                time = tiss.cellpops[p].divtime + tiss.cellpops[p].refract
                break
    return time

def supraseed(s, tau = 1000, maxseed = 1000):
    '''
        Characteristic way to generate suprabasal seeds for simulations
        
        Use results['seed'] as the maxseed if it exists
    '''
    return int(tau*maxseed + s)


def getCdf(spdf):
    '''
        Converts pdf to cdf, works provided the values are kept along last dimension,
        Assumes but does not check normalisation
    '''
    return np.cumsum(spdf[...,::-1], axis = -1)[...,::-1]   

def getPdf(sample, maxbin = None):
    '''
        Takes a sample and extracts the pdf
    '''
    if maxbin is None:
        maxbin = np.max(sample)
    
    return np.histogram(sample, bins = np.arange(0, maxbin + 2), density = True)[0]



