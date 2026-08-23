'''
    Implementation of custom metrics on datasets objects, for use in 
        metric construction in tissfit.
'''


import numpy as np
import datasets as ds
import metrics
from numba import jit


def logLL_joint(sim_data, exp_data, m = 'jointCS', use_jeffreys = False, exclude_times = None,
                use_jeff1 = True):
    '''
        Log-likelihood for a joint distribution 
            of basal and suprabasal sizes
    '''

    ntimes = sim_data.custom_quants[m].shape[0]
    subtimes = np.ones(ntimes, dtype = bool)
    
    if exclude_times is not None:
        exc = np.array(exclude_times)
        exc = exc[exc < ntimes]
        subtimes[exc] = False

    sim_jcs = sim_data.custom_quants[m][subtimes,...]
    exp_jcs = exp_data.custom_quants[m][subtimes,...]
    
    #Make sizes conform 
    if np.shape(exp_jcs) != np.shape(sim_jcs):
        new_shape = np.max([sim_jcs.shape, exp_jcs.shape], axis = 0)
        new_shape[1:] += 1 #increase dimensions along pdf axes
        nsimjcs = np.zeros(tuple(new_shape))
        nsimjcs[:sim_jcs.shape[0],:sim_jcs.shape[1],:sim_jcs.shape[2]] += sim_jcs
        nexpjcs = np.zeros(tuple(new_shape))
        nexpjcs[:exp_jcs.shape[0],:exp_jcs.shape[1],:exp_jcs.shape[2]] += exp_jcs 
    else:
        nexpjcs = exp_jcs
        nsimjcs = sim_jcs

    sim_nsamples = sim_data.nsamples[m][subtimes,...]
    exp_nsamples = exp_data.nsamples[m][subtimes,...]
    
    return metrics.log_likelihood(nsimjcs, nexpjcs, sim_nsamples, exp_nsamples, use_jeffreys, use_jeff1)



def logLL_move(sim_data = None, exp_data = None, use_basal = False, use_supra = False, use_jeffreys = True):
    '''
        Negative log-likelihood of moves, works for moves of different dimensionalities 
            (basal only or basal and suprabsal)
        
        Uses a Jeffrey's prior by default, other options not implemented
    '''
    
    if use_basal:
        allmoves_sim = sim_data.custom_quants['btranCS']
        allmoves_exp = exp_data.custom_quants['btranCS']
        #livesize_exp = exp_data.raw_data['liveCS']
        norm = np.sum(allmoves_sim, axis = -1)
        Njeff = Njeff_trans_bas(allmoves_exp)
    elif use_supra: #suprabasal layer conditioned on previous basal and current
        allmoves_sim = sim_data.custom_quants['tranCS']
        allmoves_exp = exp_data.custom_quants['tranCS']
        #livesize_exp = exp_data.raw_data['liveCS']        
        norm = np.sum(allmoves_sim, axis = -1)
        Njeff = Njeff_trans_sb_cond(allmoves_exp)
    else:
        allmoves_sim = sim_data.custom_quants['tranCS']
        allmoves_exp = exp_data.custom_quants['tranCS']
        norm = np.sum(allmoves_sim, axis = (3,4))
        Njeff = Njeff_trans_full(allmoves_exp)

    Llike_full = -allmoves_exp.T*(np.log(allmoves_sim + 0.5).T - np.log(Njeff/2 + norm).T)
    return np.sum(Llike_full, axis = tuple(np.arange(0,np.ndim(Llike_full)-1)))


def logLL_move_bas(sim_data, exp_data, use_jeffreys = True):
    ''' 
        Basal-only log-likelihood of moves
    '''
    return logLL_move(sim_data, exp_data, use_basal = True, use_jeffreys = use_jeffreys)
    
def logLL_move_sb(sim_data, exp_data, use_jeffreys = True):
    ''' 
        Suprabasal conditioned on previous state and current basal, log-likelihood of moves
    '''
    return logLL_move(sim_data, exp_data, use_supra = True, use_jeffreys = use_jeffreys)


#Custom distance functions
def logLL_move_ctype(sim_data = None, exp_data = None, allmoves_sim = None, allmoves_exp = None, 
               ctypes = None, allmoves_bas_sim = None):
    ''' 
        Negative log-likelihood of moves segmented by clone-type
        
            use ctypes = [] for all clones (default), and ['S', 'T', 'D'] 
    '''
    
    if allmoves_sim is None:
        if sim_data is not None:
            allmoves_sim     = sim_data.custom_quants['cttranCS']
            allmoves_bas_sim = sim_data.custom_quants['ctbtranCS']
        else:
            assert(0)
            
    if allmoves_exp is None:
        if exp_data is not None:
            allmoves_exp = exp_data.custom_quants['cttranCS']
        else:
            assert(0)            

    if ctypes is None:
        if sim_data is not None:
            ctypes = sim_data.data_attributes['ctypes']
        else:
            ctypes = ['S', 'T', 'D']
        
    mice = allmoves_exp['mice_ID'].unique()
    
    if allmoves_bas_sim is None:
        allmoves_bas_sim = np.sum(allmoves_sim, axis = (2,4))

    miceIDs = {mice[i]: i for i in range(len(mice))}
    cids = {ctypes[i]: i for i in range(len(ctypes))}
    ntimes = allmoves_sim.shape[0]

    Lterms = np.zeros((ntimes, len(mice), 2, len(ctypes)))
    norm_bas = np.zeros((ntimes, len(ctypes), allmoves_bas_sim.shape[1]))
    norm_sb  = np.zeros((ntimes, len(ctypes), allmoves_sim.shape[1], allmoves_sim.shape[2], allmoves_sim.shape[3])) #normalisation of suprabasal conditioned on basal

    for t in range(ntimes):
        for c in range(len(ctypes)):
            norm_bas[t,c,:]     = np.sum(allmoves_bas_sim[t,:,:,c],1)
            norm_sb[t,c,:,:,:]  = np.sum(allmoves_sim[t,:,:,:,:,c],3)

    for i in range(len(allmoves_exp)):
        row = allmoves_exp.iloc[i]
        t = row['time']
        c = cids[row['ctype']]
        m = miceIDs[row['mice_ID']]
        
        b0 = row['basal0']
        b1 = row['basal1']
        s0 = row['supra0']
        s1 = row['supra1']
        count = row['count']
        
        lDM_bas = count*np.log((allmoves_bas_sim[t,b0,b1,c] + (allmoves_bas_sim[t,b0,b1,c] == 0))/(norm_bas[t,c,b0] + 1))
        assert((allmoves_sim[t,b0,s0,b1,s1,c])/(norm_sb[t,c,b0,s0,b1] + 1) <= 1)
        lDM_bsb = count*np.log((allmoves_sim[t,b0,s0,b1,s1,c] + (allmoves_sim[t,b0,s0,b1,s1,c] == 0))/(norm_sb[t,c,b0,s0,b1] + 1))
        Lterms[t, m, 0, c] += lDM_bas
        Lterms[t, m, 1, c] += lDM_bsb

    #np.sum(Lterms)
    return -Lterms
    
#Normalisation factors for Jeffrey prior for transitions
@jit(nopython=True)
def Njeff_trans_full(allmoves_exp):
    '''   
        Calculates Normalisation constant for Jeffrey prior 
            for transition
    '''
    Njeff = np.ones(allmoves_exp.shape[:3])

    for t in range(allmoves_exp.shape[0]):
        for b in range(allmoves_exp.shape[1]):
            for s in range(allmoves_exp.shape[2]):
                if np.sum(allmoves_exp[t,b,s,:,:]) == 0:
                    continue
                maxX = [np.max(w)+1 for w in np.where(allmoves_exp[t,b,s,:,:] > 0)]
                Njeff[t,b,s] = np.prod(maxX)
    return Njeff


@jit(nopython=True)
def Njeff_trans_bas(allmoves_bas_exp):
    '''   
        Calculates Normalisation constant for Jeffrey prior 
            for transition
    '''
    Njeff = np.ones(allmoves_bas_exp.shape[:2])

    for t in range(allmoves_bas_exp.shape[0]):
        for b in range(allmoves_bas_exp.shape[1]):
            if np.sum(allmoves_bas_exp[t,b,:]) == 0:
                continue
            maxX = [np.max(w)+1 for w in np.where(allmoves_bas_exp[t,b,:] > 0)]
            Njeff[t,b] = np.prod(maxX)
    return Njeff


@jit(nopython=True)
def Njeff_trans_sb_cond(allmoves_exp):
    '''   
        Calculates Normalisation constant for Jeffrey prior 
            for suprabasal conditioned on basal
    '''
    
    #Old implementation has bad performance
    #Need to re-implement using live-size
    
    Njeff = np.ones(allmoves_exp.shape[:4])

    for t in range(allmoves_exp.shape[0]):
        for b in range(allmoves_exp.shape[1]):
            for s0 in range(allmoves_exp.shape[2]):
                for s1 in range(allmoves_exp.shape[3]):
                    if np.sum(allmoves_exp[t,b,s0,s1,:]) == 0:
                        continue
                    Njeff[t,b,s0,s1] = np.max(np.where(allmoves_exp[t,b,s0,s1,:] > 0)[0])+1
        return Njeff
