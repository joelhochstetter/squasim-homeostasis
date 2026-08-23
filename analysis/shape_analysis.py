''' 
    Analysis of cell shapes, forces and topological transitions:
        Currently code to calculate:
            - T1 transitions
            - Shape parameters
            - Mean squared displacement (including with division)
            - Effective diffusion coefficient
'''


from unittest import skip
import numpy as np
from collections import ChainMap
from scipy import stats


# T1 transitions

def array_row_difference(a, b):
    ''' 
        Rows in a that are not in b
    '''
    common_rows = (a[:, None] == b).all(-1).any(1)
    return a[~common_rows]
 
def array_row_intersection(a, b):
   return a[(a[:, None] == b).all(-1).any(1)]


def T1s_nodiv(jlengths):
    '''
        Identifies T-1 transitions in vertex model simulations without division
            using junction length matrix
    '''
    adjmats = jlengths > 0
    tsteps = len(jlengths[:,0,0]) - 1

    t1s = {}

    nt1  =  np.array(np.where((jlengths[:-1,:,:] == 0.0) & (jlengths[1:,:,:] > 0)))
    ot1 = np.array(np.where((jlengths[:-1,:,:] > 0) & (jlengths[1:,:,:] == 0)))

    newtouches = {}
    for i in range(np.shape(nt1)[1]):
        if nt1[1,i] < nt1[2,i]:
            newtouches.setdefault(nt1[0,i],[]).append(list(np.sort(nt1[1:3,i])))
        
    oldtouches = {}
    for i in range(np.shape(ot1)[1]):
        if ot1[1,i] < ot1[2,i]:
            oldtouches.setdefault(ot1[0,i],[]).append(list(np.sort(ot1[1:3,i])))    

    #What do we do with other topological changes? Either boundary or from division or loss
    times = np.sort(list(set(list(newtouches.keys())) & set(list(oldtouches.keys()))))

    for t in times:
        for nt in newtouches[t]:
            common_nt = list(np.sort(list(set(np.where(adjmats[t,nt[0],:])[0]) & set(np.where(adjmats[t+1,nt[0],:])[0]) & set(np.where(adjmats[t,nt[1],:])[0]) & set(np.where(adjmats[t,nt[1],:])[0]))))
            if common_nt in oldtouches[t]:
                t1s.setdefault(t, []).append((nt, common_nt))
                
                
def T1s(jlengths, cellids):
    '''
        Identifies T-1 transitions in vertex model simulations with or without division
            using juncion length matrix
    '''
    
    adjmats = [jad > 0 for jad in jlengths]
    tsteps = len(cellids) - 1    

    t1s = {}

    for t in range(tsteps):
        jt  = cellids[t][np.array(np.where(jlengths[t] > 0)).transpose()]
        jt1 = cellids[t+1][np.array(np.where(jlengths[t+1] > 0)).transpose()]
        
        adjmat_t  = adjmats[t]
        adjmat_t1 = adjmats[t+1]
        
        inverseid_t  = dict(zip(cellids[t], np.arange(len(jlengths[t]))))
        inverseid_t1 = dict(zip(cellids[t+1], np.arange(len(jlengths[t+1]))))
        
        
        old_touches = array_row_difference(jt,jt1)
        new_touches = array_row_difference(jt1,jt)
        
        #extract only unique (row < columns):
        old_touches = old_touches[(old_touches[:, 0] < old_touches[:, 1])]
        new_touches = new_touches[(new_touches[:, 0] < new_touches[:, 1])]
        
        for nt in new_touches:
            if nt[0] in cellids[t] and nt[0] in cellids[t+1] and nt[1] in cellids[t] and nt[1] in cellids[t+1]:
                nt0 = np.array([inverseid_t[x] for x in nt])
                nt1 = np.array([inverseid_t1[x]for x in nt])
                common_nt = list(np.sort(list(set(cellids[t][np.where(adjmat_t[nt0[0],:])[0]]) & set(cellids[t+1][np.where(adjmat_t1[nt1[0],:])[0]]) & set(cellids[t][np.where(adjmat_t[nt0[1],:])[0]]) & set(cellids[t+1][np.where(adjmat_t1[nt1[1],:])[0]]))))
                if len(common_nt) == 2 and common_nt in old_touches:
                    t1s.setdefault(t, []).append((list(nt), common_nt))
                    
    return t1s


def num_T1s(jlengths, cellids):
    '''
        Returns the nunber of T1 transitions in a simulation at each time-point
    '''
    all_t1s = T1s(jlengths, cellids)
    nt1s = {tx: len(all_t1s[tx]) for tx in all_t1s.keys()}
    
    numt1 = np.zeros(len(cellids))
    
    if len(nt1s.keys()) > 0:
        numt1[np.array(list(nt1s.keys()))] += np.array(list(nt1s.values()))
    
    return numt1


def num_4f(jlengths, perims = None, cutoff = 0.05, norm_by_perim = True):
    if perims is None:    
        perims = [np.sum(jac,1) for jac in jlengths]
    av_perims = [np.mean(p) for p in perims]
    if norm_by_perim:
        return [np.sum((jlengths[t]/av_perims[t] < cutoff) & (jlengths[t] > 0))//2  for t in range(len(jlengths))]
    else:
        return [np.sum((jlengths[t] < cutoff) & (jlengths[t] > 0))//2  for t in range(len(jlengths))]        


def p0(areas, perims = None, jlengths = None):
    if perims is None:
        perims = [np.sum(jac,1) for jac in jlengths]
    return [perims[t]/np.sqrt(np.abs(areas[t])) for t in range(len(areas))] 


def pop_areas(areas, ncells, skip):
    npops = ncells.shape[1]
    splits = np.hstack([np.zeros((len(ncells),1), dtype = int), np.cumsum(ncells, axis = 1)])[::skip,:]
    pop_areas = [[np.mean(np.abs(areas[t][splits[t,p]:splits[t,p+1]])) for p in range(npops)] for t in range(len(areas))]
    return pop_areas

def msddiv(ncells, ancs, cellid, cellpos, skip):
    
    if type(ancs) is list:
        ancs  = dict(ChainMap(*ancs))
        
    msdcell = np.zeros(int(len(ncells)/skip) + 1)
    t0indices = {cellid[0][c]: c for c in range(len(cellid[0]))}

    for t in range(0, int(len(ncells)/skip) + 1):
        ncell = np.sum(ncells[t*skip])
        cellpost = cellpos[t][:ncell] #excludes suprabasal cells
        ancpercell = np.array([t0indices[ancs[cid]] for cid in cellid[t][:ncell]]) #excludes suprabasal cells
        msdcell[t] = np.mean(np.sum(np.square(cellpost - cellpos[0][ancpercell]),1))

    return msdcell
