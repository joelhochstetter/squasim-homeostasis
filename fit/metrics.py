'''
    This calculates metrics from datasets objects
'''


import numpy as np
import scipy.stats
import copy
import datasets as ds


dist_metrics = ['lsq', 'kld', 'ksd', 'lLL']
qmetrics = ['lsq', 'pls', 'els', 'lLL', 'lLLnb']


shortname = {'basalCS': 'bas', 'totalCS': 'tot', 'supraCS': 'sup', 'density': 'dens', 
             'divrate': 'dr', 'pers': 'pers', 'lcd': 'lcd', 'death': 'dth'}


class metric():
    '''
        Define a custom metric, which could generically depend on different pdfs
            or time-varying quantities
            
        Specify a metric as the type of data averages
            metric = {use_sum = True %(use sum of metric at datapoints, else use sum of log(metric), previously called cost_sum, cost_prd),
                    quantity = [] %choose from 'basalCS' / 'totalCS' / 'density' / 'divrate',
                    dist = [] % choose from 'basalCS' / 'supraCS' / 'totalCS',
                    dist_metric = 'lsq' / 'kld' z/ 'ksd' / 'cvM' (cramer von Mieses, least squares of cdf) / 'lL' (negative log-likelihood),
                    popOnly = -1 (use any population) / 0/1/2: specify population 
                            /[1,2]: specify multiple populations,
                    popnames = -1 (use number), or list of names for populations, e.g. ['D', 'T', 'S'],
                    qmetric = 'lsq' / 'pls' / 'els' (for quantity use least squares or percentage least squares or 
                        least squares normalised by the standard error)
                    exclude_times = {}  (indices of the times to to exclude),
                    norm_t0 = [] (sets quantities to normalise to 1 at t0 for comparison, 
                        e.g. for LCD or persistence),
                    av_time = False (if True, then the metric is averaged over time, else use a sum),
                    weights = {} (weights for each quantity/distribution in metric, defaults to 1)
                    custom_metrics = {} (specify the functions needed to compute custom mtrics),
                    kwargs = {} (kwargs for custom metrics or dist metrics),
                    }  
                                                           
    '''
    def __init__(self, quantity = [], dist = [], 
                 qmetric = 'lsq', dist_metric = 'lsq', 
                 popOnly = -1, popnames = None, exclude_times = None,
                 av_time = False, weights = None, use_sum = True,
                 custom_metrics = None, mindiff = 1e-8, kwargs = None, 
                 name = None, cost = None):
        

        self.quantity = quantity
        self.dist = dist
        self.dist_metric = dist_metric
        self.qmetric = qmetric
        self.popOnly = popOnly
        
        if weights is None:
            self._weights = {}
        else:
            self._weights = weights 
        self.use_sum = use_sum      
              
         
        if self.popOnly != -1:
            if np.ndim(self.popOnly) == 0:
                self.popOnly = [self.popOnly] 
            if len(self.popOnly) == 0:
                self.popOnly = -1
        
        self.sub_metrics  = {} #contains the sub-metrics as scalars
        self.full_metrics = {} #contains the full sub-metrics as arrays, temporal variation, etc
        
        self.av_time = av_time
        
        #set-up exclude times
        if exclude_times is None:
            self.exclude_times = {}
        else:
            self.exclude_times = exclude_times
        
        #custom mtrics
        if custom_metrics is None:
            self.custom_metrics = {}
        else:
            self.custom_metrics = custom_metrics 
        
        self.setup_kwargs(kwargs)
        
        
        if self.use_sum is True:
            mindiff = 0
        self.mindiff = mindiff #minimum value of metric when using product of costs
        
        if name is None:
            self.name = self.metname(popnames)
        else:
            self.name = name


    def get_kwargs(self, d):
        '''
            Extract kwargs for a quantity or distribution for use in metric calculation

        '''
        if self.kwargs is None or d not in self.kwargs:
            return {}
        else:
            return self.kwargs[d]
        

    def setup_kwargs(self, kwargs):
        '''
            Set-up kwargs
        '''

        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs

        for cm in self.custom_metrics.keys():
            if cm in self.exclude_times:
                if cm not in self.kwargs:
                    self.kwargs[cm] = {}
                self.kwargs[cm]['exclude_times'] = self.exclude_times[cm]

        
        
    def weights(self, d):
        '''
            Extract weights for a quantity or distribution
        '''
        if d not in self._weights:
            return 1
        else:
            return self._weights[d]
        
    def allquantities(self):
        '''
            Returns names of all quantities and distributions
        '''
        return self.quantity + self.dist + list(self.custom_metrics.keys())
        
    def copy(self):
        return copy.deepcopy(self)        
        
    def metname(self, popnames = None):
        if len(self.dist) + len(self.quantity) > 1:
            if self.use_sum:
                mname = 'cost_sum'
            else:
                mname = 'cost_prd'
        else:
            mname = 'cost'
        
        if len(self.dist) > 0:
            if self.dist_metric != 'lsq':
                mname += '_' + self.dist_metric
            else:
                mname += '_lsqd'
            for d in sorted(list(self.dist)):
                mname += '_' + self.qsavename(d)
            
        if len(self.quantity) > 0:
            mname += '_av'
            if self.qmetric != 'lsq':
                mname += '_' + self.qmetric
            for q in sorted(list(self.quantity)):
                mname += '_' + self.qsavename(q)
            
        #this won't work correctly
        if self.popOnly != -1:
            mname += '_'
            if popnames is None:
                for p in self.popOnly:
                    mname += p
                else:
                    for p in self.popOnly:
                        mname += popnames[p]  
                             
        if len(self.exclude_times) > 0:
            mname += '_excld'
            for k in self.exclude_times:
                for t in self.exclude_times[k]:
                    mname += '_' + str(t)
                                         
        return mname
    
    
    def qsavename(self, q):
        '''
            Returns name of quantity
        '''
        if q in shortname:
            return shortname[q]
        else:
            return q
         
         
    def conform_dist_dims(self, data1, data2):
        '''
            Make distribution dimensions conform
        '''
        # Fix max pdf bins to actual values
        data1.fix_max_pdf_bin()
        data2.fix_max_pdf_bin()

        new_max = {key: max(data1.max_pdf_bin[key], data2.max_pdf_bin[key]) for key in data1.max_pdf_bin if key in data2.max_pdf_bin}
        data1.reshape_pdfs(dists_to_reshape = list(new_max.keys()), max_bin = new_max)
        data2.reshape_pdfs(dists_to_reshape = list(new_max.keys()), max_bin = new_max)

        assert(np.all([np.shape(data1.dists[d]) == np.shape(data2.dists[d]) for d in data1.dists if d in data2.dists]))

        
    def calc(self, data1 = None, data2 = None, 
             quants1 = None, quants2 = None, dists1 = None, dists2 = None):
        '''
            Calculate the metric on two datasets
            
            By default: data1 is simulated data, data2 is experimental data
        '''

        #Get data
        if data1 is not None:
            quants1 = data1.quantities
            dists1  = data1.dists
            se_quants1 = data1.se_quants
            nsamples1 = data1.nsamples

        if data2 is not None:
            quants2 = data2.quantities        
            dists2  = data2.dists
            se_quants2 = data2.se_quants
            nsamples2 = data2.nsamples
            
        # Make distributional data conform
        self.conform_dist_dims(data1, data2)

        #Process data
        simAvgs = [self.get_av_data(quants1,  q) for q in self.quantity]
        expAvgs = [self.get_av_data(quants2,  q) for q in self.quantity]
        simPdfs = [self.get_dist_data(dists1, d) for d in self.dist]
        expPdfs = [self.get_dist_data(dists2, d) for d in self.dist]
        nsamples1 = {q: self.get_av_data(nsamples1, q) for q in data1.nsamples.keys()}
        nsamples2 = {q: self.get_av_data(nsamples2, q) for q in data2.nsamples.keys()}
        
        if se_quants2 is not None:
            assert(se_quants2 is not None)
            assert(len(quants2) == len(se_quants2)) 
            simSEs = None
            #simSEs = [self.get_av_data(se_quants1, q) for q in self.quantity]
            expSEs = [self.get_av_data(se_quants2, q) for q in self.quantity]
        else:
            simSEs = None
            expSEs = None
            
        #Compute metrics
        self.compute_q_metrics(simAvgs, expAvgs, self.qmetric, simSEs, expSEs,
                               sim_nsamples = nsamples1, exp_nsamples = nsamples2)
        self.compute_d_metrics(simPdfs, expPdfs, self.dist_metric, nsamples1, nsamples2)
        self.compute_custom_metrics(data1, data2)
        
        return self.cost_sum()

   
    def get_av_data(self, data, q):
        '''
            Processes average data to account for excluded times
        '''
        if q in self.exclude_times:
            included_mask = np.ones(len(data[q]), dtype=bool)
            exc = np.array(self.exclude_times[q])
            exc = exc[exc < len(data[q])]
            included_mask[exc] = False
            return data[q][included_mask]
        else:        
            return data[q]
       
        
    def get_dist_data(self, data, d):
        '''
            Processes distribution data to account for excluded times
                and smoothing
        '''
        
        
        
        if d in self.exclude_times:
            included_mask = np.ones(data[d].shape[0], dtype=bool)
            exclude_times = np.array(self.exclude_times[d])
            included_mask[exclude_times[exclude_times < len(data[d])]] = False
            return data[d][included_mask,:]
        else:        
            return data[d]
        
     
    def exclude_quants(self, excluded):
        '''
            Creates a copy of the same metric with certain quantities excluded
        '''
         
        if np.ndim(excluded) == 0:
            excluded = [excluded]
        
        newmet = self.copy()
        
        newmet.quantity = [x for x in newmet.quantity if x not in excluded]
        newmet.dist     = [x for x in newmet.dist     if x not in excluded]
        newmet.name = newmet.name()
        
        return newmet
        

    def compute_q_metrics(self, simAvgs = [], expAvgs = [], qmetric = 'lsq', simSEs = None, expSEs = None, 
                          use_sum = True, mindiff = 1e-8, sim_nsamples = None, exp_nsamples = None):
        '''
            Compute the metric for each quantity
        '''
        
        assert(len(simAvgs) == len(expAvgs))        
        
        for i in range(len(simAvgs)):
            #Replace equalities with minimum difference
            name = self.quantity[i]
            d = quant_dist(simAvgs[i], expAvgs[i], qmetric, get_ith(simSEs, i), get_ith(expSEs, i),
                           get_ith(sim_nsamples, name), get_ith(exp_nsamples, name))
            self.full_metrics['q' + name] = d
            
            #Combine over time
            self.sub_metrics['q' + name]  = self.combine_times(d)

    
    
    def compute_d_metrics(self, simPdfs = [], expPdfs = [], dist_metric = 'lsq', sim_nsamples = None,
                          exp_nsamples = None):
        ''' 
            Compute the metric for each distribution
            
            sum_dist uses sum of metric at datapoints, else use sum of log(metric)
        '''
        for i in range(len(simPdfs)):
            name = self.dist[i]
            d = dist_dist(simPdfs[i], expPdfs[i], dist_metric, get_ith(sim_nsamples, name), get_ith(exp_nsamples, name), self.get_kwargs(name))
            self.full_metrics['d' + name] = d
            self.sub_metrics['d' + name]  = self.combine_times(d)

        
    def combine_times(self, d):
        '''
            Combines a metric over time
        '''
        if self.use_sum is False:
            d[d < self.mindiff] = self.mindiff
            d = np.log(d)

        if self.av_time:
            return np.mean(d)
        else:
            return np.sum(d)
        
        
    def compute_custom_metrics(self, data1, data2):
        '''
            Compute any custom metrics from data1 and data2
            
            Any custom metric
        '''
        
        for m in self.custom_metrics:
            self.full_metrics[m] = self.custom_metrics[m](data1, data2, m, **self.get_kwargs(m))
            self.sub_metrics[m]  = self.combine_times(self.full_metrics[m])
            
            

    def cost_sum(self):
        '''
            Calculates cost function, combining across sub-metrics applying the specified weights
        '''       
        
        cost = 0
        
        for m in self.sub_metrics:
            cost += self.sub_metrics[m]*self.weights(m)
        
        return cost




def quant_dist(simAvg, expAvg, qmetric = 'lsq', simSE = None, expSE = None,
               sim_nsamples = None, exp_nsamples = None):

    '''
        Distance metric between quantities
            simAvg, expAvg: are given as L length array
            qmetric: 'lsq'/'pls' (for quantity use least squares or percentage least squares)
            / 'els' (for quantity use least squares normalised by the standard error)
            / 'lLL' (log likelihood assuming normal)
            / 'lLLnb' (log likelihood assuming negative binomial distribution)
    '''
    
    assert(qmetric in qmetrics)
    
    if qmetric == 'lsq':
        qm = np.square(simAvg - expAvg)
    elif qmetric == 'pls':
        qm = np.square(simAvg - expAvg)/(np.square(expAvg) + (expAvg == 0)/exp_nsamples)
    elif qmetric == 'els':
        qm = np.square(simAvg - expAvg)/(np.square(expSE) + (expSE == 0)/exp_nsamples)
    elif qmetric == 'lLL':
        qm = -scipy.stats.norm.logpdf(expAvg, simAvg, expSE)
    elif qmetric == 'lLLnb':
        qm = llike_nbinom(simAvg, expAvg, exp_nsamples)

    return qm
    
    

    

def llike_nbinom(simAvg, expAvg, exp_nsamples):
    '''
        Log likelihood of negative binomial distribution
    '''
    
    return -scipy.stats.nbinom.logpmf(((expAvg)*exp_nsamples).astype(int), exp_nsamples, 1/(1+simAvg))
    
    
def dist_dist(simPdf, expPdf, dist_metric = 'lsq', 
                sim_nsamples = None, exp_nsamples = None, met_kwargs = None):
    ''' 
        Distance metric between distributions
            Pdfs: are given as TxL array, where T is time, and L is number of bins, or an L length array (no time) 
            Distance metric between distributions: 'lsq'/'kld'/'ksd'/'lLL'            
            sim_nsamples, exp_nsamples: are the number of samples for each time-point needed
                for some metrics
    '''
    
    assert(dist_metric in dist_metrics)
    
    if met_kwargs is None:
        met_kwargs = {}
    
    if   dist_metric == 'lsq':
        dis = lsq_dist(expPdf, simPdf)
    elif dist_metric == 'ksd':
        dis = KSD(getCdf(expPdf), getCdf(simPdf))
    elif dist_metric == 'kld':
        dis = kld(expPdf, simPdf)
    elif dist_metric == 'lLL':
        dis = log_likelihood(simPdf, expPdf, sim_nsamples, exp_nsamples, **met_kwargs)
        
    return dis

        
    
def lsq_dist(exppdf, simpdf):
    '''
        Input distributions as TxL array, where T is time, and L is number of bins
        Returns lsq for each time-point
    '''     
    return np.sum(np.square((exppdf - simpdf)),-1)
    
    
def getCdf(spdf):
    '''
        Converts pdf to cdf, works provided the values are kept along last dimension,
        Assumes but does not check normalisation
    '''
    return np.cumsum(spdf[...,::-1], axis = -1)[...,::-1]   


def KSD(cdf1, cdf2):
    '''
        Calculates Kolmogorov Smirnov distance from cdfs
        Bins must be the last index so can work across multiple time-points
    '''
    return np.max(np.abs(cdf1 - cdf2),-1)



def kld(exppdf, simpdf):
    '''
        Calculates Kullback-Leibler divergence from pdfs
        Bins must be the last index so can work across multiple time-points
    '''
    
    
    return scipy.stats.entropy(exppdf[...,1:], simpdf[...,1:], axis = -1)

#weighted average mean:
def wam(vector, weights):
    assert(np.shape(vector)[1] == len(weights))
    scale = vector.copy()
    for i in range(len(weights)):
        scale[:,i] = scale[:,i]*weights[i]
    
    return np.sum(scale, 1)/np.sum(weights)
    
    
    
def jeffreys_smooth(sim_dist, nsamples, exp_dist = None):
    '''  
        Apply Jeffrey's smoothing to ensure that that there are no 
            zeroes for calculating the like-lihood function
            
        sim_dist, exp_dist: are one or multi-dimensional distributions
    '''
    
    if exp_dist is None or np.shape(exp_dist) != np.shape(sim_dist):
        new_dist = getcounts(sim_dist, nsamples) + 0.5
    else:
        maxX = np.array([[np.max(w) for w in np.where(exp_dist[t,...] > 0)] for t in range(sim_dist.shape[0])])
        new_dist = getcounts(sim_dist, nsamples)
        
        for t in range(new_dist.shape[0]):
            slices = tuple(slice(None, 1+maxX[t, d]) for d in range(maxX.shape[1]))
            new_dist[(t,) + slices] += 0.5

    norms = np.sum(new_dist, axis = tuple(np.arange(1, np.ndim(sim_dist))), keepdims = True)
    
    return new_dist/norms
    
def getcounts(pdf, nsamples):
    return pdf * nsamples.reshape(-1, *(1,) * (pdf.ndim - 1))
    
    
def is_supra(met):
    '''
        Returns if a metric contains terms for the suprabasal layer
    '''
    return (any(item in ['supraCS', 'totalCS'] for item in met.quantity) or any(item in ['supraCS', 'totalCS'] for item in met.dist))



def log_likelihood(sim_dist, exp_dist, sim_nsamples, exp_nsamples, use_jeffreys = False, jeff1 = True):
    '''
        Compute log-likelihood of a distribution with an option to use Jeffrey's prior
            based on the maximum experimental clone-size at each time-point
        
        Will work for single or multidimensional distributions
        
        Returns the terms in the log-likelihood for each time-point

        jeff1: if True consider probability of events that don't occur as 1/N
    '''
    
    fm = exp_dist.T*exp_nsamples
    
    if use_jeffreys:
        maxX = np.array([[np.max(w) if len(w) > 0 else 0 for w in np.where(exp_dist[t,...] > 0)] for t in range(sim_dist.shape[0])])
        Njeff = np.prod(maxX + 1, axis = tuple(np.arange(1, np.ndim(maxX))))/2

        return -np.sum((fm*(np.log(sim_dist.T*sim_nsamples + 0.5) - np.log(sim_nsamples + Njeff))), axis = tuple(np.arange(0, np.ndim(exp_dist)-1)))
    elif jeff1:
        return -np.sum(fm*np.log(sim_dist.T*sim_nsamples/(sim_nsamples+1) + 1*(sim_dist.T == 0)/(sim_nsamples+1)), axis = tuple(np.arange(0, np.ndim(exp_dist)-1)))
    else:
        return -np.sum(fm.T*np.log(sim_dist + 1e-8*(sim_dist == 0)), axis = tuple(np.arange(0, np.ndim(exp_dist)-1)))





def get_ith(x, i):
    ''' 
        Returns the ith element of a list else returns the None
    '''
    if x is not None and type(x) is list and i < len(x):
        return x[i]
    elif x is not None and type(x) is dict and i in x:
        return x[i]
    else:
        return None
    
 