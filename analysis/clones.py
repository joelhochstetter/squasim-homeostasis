'''
    Analysis pipeline for clones from simulations and experiments,
        we can handle static or time-lapse data. 
        
        Allows quick calculations of different quantities of interest 
        and importantly we don't calculate more quantities than we need, 
        which speeds up fitting and analysis.
'''


import numpy as np
import matplotlib.pyplot as plt
import analysis
import pandas as pd
import copy
import scipy
import sparse
from numba import jit, njit
from types import FunctionType



class lanalysis():
    def __init__(self, livesize, popnames = None, tvals = None, containsBasal = True, suprabasal = True,
                 miss_flag = -100, ctypes = [], repair_miscounts = True,
                 repair_ext_missing = False, t0 = 0, subpop = None, copy_livesize = True,
                 use_t0 = True, tscale = 1, verbose = True):
        '''
            Analysis of clones from simulations or experiments given a live-size is provided,
                or static clone sizes (currently not implemented)   
                
            Assumes no data is missing
            Missing data is denoted by clone-sizes smaller than 0 in both categories
                
            Flags:
                containsBasal, set to True to only include clones that contain basal cells
                suprabasal, set to True to include suprabasal cells in the analysis
                    Currently coded to only work with one supra-basal layer  
                    
                repair_miscounts: accounts for mistakes in the counting process
                
                t0: removes clones with zero basal cells at t = 0
        '''
        self.inducttime = 0
        self.use_t0 = use_t0
        self.verbose = verbose
        
        #Calculate the live-size of each clone, all other quantities are calculated on the fly
        if livesize is not None:
            if copy_livesize:
                self.livesize = livesize.copy()
            else:
                self.livesize = livesize

            #Set times
            self._ntimes = None
            if tvals is None:
                self.tvals = np.arange(self.ntimes)/tscale
            else:
                self.tvals = np.array(tvals)/tscale
                    
            if suprabasal:
                self.npops = livesize.shape[2] - 1
            else:
                self.npops = livesize.shape[2]
                
            assert(self.npops > 0)
                
            if np.min(livesize) < 0:
                if self.verbose:
                    print('Warning: Dataset contains missing data')
                #Ensure only data which has values in each compartment is included
                livesize[np.min(livesize, 2) < 0, :] = miss_flag    

        else:
            self.livesize = None
            
          
        #Used during time-cropping, for simulations
        self.full_tvals = None
        self.full_livesize = None
        self.timecropped = False
            
        #Flag for calculating quantities
        self.containsBasal = containsBasal
        self.suprabasal = suprabasal
        
        #Cell-types defined based on future lineage behaviour
        self.ctypes = ctypes
        self._popnames = popnames  
        
        #set properties to initial values
        self.reset_properties()
        
        #Repair data
        if self.livesize is not None:
            self.repair_data(repair_miscounts, repair_ext_missing, t0)
            self.subpopulation(subpop)


    @property
    def nclones(self):
        if self._nclones is None and self.livesize is not None:
            self._nclones = self.livesize.shape[0]
        return self._nclones
    

    @property
    def ntimes(self):
        if self._ntimes is None:
            if self._ntimes is None and self.livesize is not None:
                self._ntimes = self.livesize.shape[1]
        return self._ntimes
    
    @ntimes.setter
    def ntimes(self, value):
        self._ntimes = value
    

    def repair_data(self, repair_miscounts = True, repair_ext_missing = False, t0 = 0):
        '''
            Repairs data before performing analysis
        '''
        
        if repair_miscounts:
            self.repair_znz_miscounts()
            
        if repair_ext_missing:
            self.repair_extinct_then_missing()
            
        if self.use_t0:
            self = self.sur_t0(t0, use_copy = False)
            
        self.set_nans_negative()

    def set_nans_negative(self):
        if self.livesize is not None:
            self.livesize[np.isnan(self.livesize)] = -100

    def reset_properties(self):
        #Re-compute key properties
        if self.livesize is not None:
            if self.suprabasal:
                self.nsb0 = np.sum(self.livesize[:,0,-1]*(self.livesize[:,0,-1] > 0)) #number of suprabasal cells at t =0
        
        #Key properties
        self._nclones = None
        self._ntimes  = None

        #Base quantities
        self._joint_pdf = None
        self._joint_pdf_sparse = None
        self._joint_bas_sb = None
        self._live_bas_sb = None
        self._bas_ls = None 
        self._tot_ls = None
        self._sup_ls = None
        
        self._nsheds = None #Number of definitely shed cells
        self._max_tot = None #Maximum total number of cells accounting for shed cells
        self._tot_w_sheds = None #Total number of cells, accounting for cells shed at earlier time-points
        
        self._nsur = None #Number of surviving clones
        self._pers = None #Fraction of surviving clones
        
        self._av_bas_cs = None #Average number of basal cells per clone
        self._av_tot_cs = None #Average number of total cells per clone
        self._av_sup_cs = None #Average number of suprabasal cells per clone
        
        self._se_bas_cs = None #Standard error of basal cells per clone
        self._se_tot_cs = None #Standard error of total cells per clone
        self._se_sup_cs = None #Standard error of suprabasal cells per clone
        self._se_pers   = None #Standard error of the fraction of surviving clones
        self._se_lcd    = None #Standard error of the labelled cell density
        
        self._ev_bas_cs = None #Even bias for basal clone sizes
        self._ev_tot_cs = None #Even bias for total clone sizes
        self._ev_sup_cs = None #Even bias for suprabasal clone sizes
        
        self._bas_pdf = None #PDF of basal cells per clone
        self._tot_pdf = None #PDF of total cells per clone
        self._sup_pdf = None #PDF of suprabasal cells per clone
        self._lcd = None #labelled cell density
        self._missing = None #Number of missing clones per time-point
        
        self._full_joint_pdf = None #Full joint pdf of clone sizes
        self._full_bas_pdf = None #Full pdf of basal cells per clone
        self._full_tot_pdf = None #Full pdf of total cells per clone
        self._full_sup_pdf = None #Full pdf of suprabasal cells per clone
        self._full_joint_bas_sb = None #Full joint pdf of clone sizes

        #Population quantities
        self._splits = None #Splits between different population indices
        self._popsur = None #Survival probability of each population
        self._pop_ave_bas_cs = None #Average number of basal cells per clone labelled by a certain population
        self._pop_ave_tot_cs = None #Average number of total cells per clone labelled by a certain population        
        self._pop_ave_sup_cs = None #Average number of suprabasal cells per clone labelled by a certain population
        self._pop_bas_pdf = None #PDF of basal cells per clone labelled by a certain population
        self._pop_tot_pdf = None #PDF of total cells per clone labelled by a certain population
        self._pop_sup_pdf = None #PDF of suprabasal cells per clone labelled by a certain population
        self._ave_cs_by_pop = None # Average cells of type per clone
        self._pop_ave_cs_by_pop = None # Average cells of type per clone labelled by a certain population


    def add_timezero(self, t0 = 0):
        '''
            Adds the initial condition of clones having 1 b 0 sb at t = 0
        '''
        
        livesize = self.livesize
        self.livesize = np.zeros((livesize.shape[0], livesize.shape[1] + 1, livesize.shape[2]), dtype = 'int32')
        self.livesize[:,1:,:] = livesize
        self.livesize[:,0,0] = 1
        self.ntimes += 1
        self.tvals = np.hstack((t0, self.tvals))
        
        self.reset_properties()
        
        return self
        
    def timecrop(self, skip = None):
        ''' 
            Re-calculates clonal dynamics for a new set of tvals
                based on skipping to have only every n elements
        '''
        
        if skip is None:
            if self.timecropped == True: #revert to uncroppd
                self.tvals = self.full_tvals
                self.livesize = self.full_livesize
            else:
                return
        else:
            if self.timecropped == False:
                self.full_tvals = self.tvals.copy()
                self.full_livesize = self.livesize.copy()
            self.timecropped = True                
            self.tvals = self.full_tvals[::skip]
            self.livesize = self.full_livesize[:,::skip,:]
            
        self.reset_properties()
        

    def extinct(self, tvals = None):
        ''' 
            Return a boolean array clones are extinct, 
                assumes that after basal cell count drops
                to zero it cannot increase
        '''
        if tvals is None:
            return np.sum(self.bas_ls == 0, 1) > 0
        else:
            return np.sum(self.bas_ls[:, tvals] == 0, 1) > 0

    def nextinct(self, tvals = None):
        '''
            Return the number of extinct clones
        '''
        return np.sum(self.extinct(tvals))


    def xfrac(self, tvals = None, max_cs = None):
        ''' 
            Fraction of extinct clones of given size
            Returns probabilities
        '''
        max_tot = self.max_tot(tvals)
        ext = self.extinct(tvals)
        if max_cs is None:
            max_cs = max_tot.max()

        return np.array([np.sum(ext & (max_tot == c))/np.sum(ext) for c in range(max_cs + 1)])


        
    def ctype_ids(self, ctype, livesize = None, tvals = None):
        ''' 
            Extract the ids of clones of a given type
        '''
        
        if livesize is None:
            livesize = self.livesize
        
        maxtot = self.max_tot(tvals)
        
        if ctype == 'S':
            return np.where(maxtot >= 3)[0]
        elif ctype == 'T':
            return np.where(maxtot == 2)[0]
        elif ctype == 'D':
            return np.where(maxtot == 1)[0]
        elif ctype == 'S2':
            return np.where(maxtot >= 2)[0]
        else:
            return np.array([])
          
    @property
    def tot_w_sheds(self):
        '''
            Returns the total number of cells, accounting for cells 
                shed at earlier time-points
        '''
        
        if self._tot_w_sheds is None:
            self._tot_w_sheds = self.tot_ls + np.cumsum(self.nsheds,1)
            
        return self._tot_w_sheds
        
        

    @property
    def nsheds(self):
        '''
            Number of definitely shed cells, accounting for missing data
            
            Time-point is counted as the the time once (after) the cell is shed
            
            In the case of missing data, the shed is assigned at the time-point
                on the data is missing
        '''
        if self._nsheds is None:
            #Correct for missing data
            if np.min(self.livesize) < 0:
                tots = miss_corrected(self.tot_ls.copy())
                sups = miss_corrected(self.sup_ls.copy())
            else:
                tots = self.tot_ls
                sups = self.sup_ls
            dtot = tots[:,1:] - tots[:,:-1]
            dsup = sups[:,1:] - sups[:,:-1]
            self._nsheds = np.zeros(tots.shape, dtype = 'int32')
            nonmissing = tots[:,1:] > 0
            self._nsheds[:,1:] = nonmissing*((dtot < 0)*(-dtot) + (dtot >= 0)*(dsup < 0)*(-dsup))

        return self._nsheds
        
    @property
    def missing(self):
        ''' 
            Number of missing clones per time-point
        '''
        if self._missing is None:
            self._missing = np.sum(np.min(self.livesize, 2) < 0, 0)
            
        return self._missing

    @property
    def non_missing(self):
        ''' 
            Number of non-missing clones per time-point
        '''
        return self.nclones - self.missing
    
    @property
    def nsamples(self):
        ''' 
            Returns number of clonal samples
        '''

        if self.containsBasal:
            return self.nsur
        else:
            return self.non_missing

    @property
    def popnames(self):
        '''
            Population names, including suprabasal
        '''
        if self._popnames is None:
            self._popnames = list(np.arange(self.npops)) + ['sb']
        else:
            if len(self._popnames) == self.npops:
                self._popnames.append('sb')
                
        return self._popnames

    @property
    def joint_pdf(self):
        '''
            Returns the joint pdf of the clone sizes
        '''
        if self._joint_pdf is None:
            self._joint_pdf = analysis.joint_pdf(self.livesize, basal_cont = self.containsBasal)

        return self._joint_pdf
    
    @property
    def joint_pdf_sparse(self):
        '''
            Returns the joint pdf of the clone sizes
        '''
        if self._joint_pdf_sparse is None:
            self._joint_pdf_sparse = analysis.joint_pdf(self.livesize, basal_cont = self.containsBasal, sparse = True)

        return self._joint_pdf_sparse
    
    @property
    def full_joint_pdf(self):
        '''
            Returns the full joint pdf of the clone sizes, including all clones, not 
                just surviving
        '''
        if self._full_joint_pdf is None:
            self._full_joint_pdf = analysis.joint_pdf(self.livesize, basal_cont = False)
        
        return self._full_joint_pdf
        
    
    @property
    def bas_ls(self):
        '''
            Returns the live number of basal cells per clone
        '''
        if self._bas_ls is None:
            self._bas_ls = np.sum(self.livesize[:,:,:self.npops],2)
            
        return self._bas_ls
    
    @property
    def tot_ls(self):
        '''
            Returns the live number of total cells per clone
        '''
        if self._tot_ls is None:
            self._tot_ls = np.sum(self.livesize, 2)
            
        return self._tot_ls
    
    @property
    def sup_ls(self):
        '''
            Returns the live number of suprabasal cells per clone
        '''
        if self._sup_ls is None:
            self._sup_ls = np.sum(self.livesize[:,:,self.npops:],2)
            
        return self._sup_ls
    
    @property
    def live_bas_sb(self):
        '''
            Returns the number of basal and suprabasal cells per clone
        '''
        
        #There are no hidden populations stored
        if np.shape(self.livesize)[2] == 2:
            return self.livesize
        
        if self._live_bas_sb is None:
            self._live_bas_sb = np.zeros((self.nclones, self.ntimes, 2), dtype = 'int32') #basal and supra-basal livesize
            self._live_bas_sb[:,:,0] = self.bas_ls
            self._live_bas_sb[:,:,1] = self.sup_ls
        
        return self._live_bas_sb
    
    
    @property
    def joint_bas_sb(self):
        '''
            Returns the joint pdf of the clone sizes
        '''

        if self._joint_bas_sb is None:
            if self.npops > 1:
                self._joint_bas_sb = analysis.joint_pdf(self.live_bas_sb, basal_cont = self.containsBasal)
            else:
                self._joint_bas_sb = self.joint_pdf
            
        return self._joint_bas_sb
    
    @property
    def full_joint_bas_sb(self):
        '''
            Returns the full joint pdf of the clone sizes, including all clones, not 
                just surviving
        '''
        if self._full_joint_bas_sb is None:
            if self.npops > 1:
                self._full_joint_bas_sb = analysis.joint_pdf(self.live_bas_sb, basal_cont = False)
            else:
                self._full_joint_bas_sb = self.full_joint_pdf

        return self._full_joint_bas_sb
    
    
    @property
    def nsur(self):
        '''
            Returns the number of surviving clones
        '''
        if self._nsur is None:
            if np.max(self.missing) == 0:
                self._nsur = np.sum(self.bas_ls > 0, 0)
            else:
                self._nsur = np.sum((self.bas_ls > 0) & (self.sup_ls >= 0), 0)
            
        return self._nsur
    
    @property
    def pers(self):
        '''
            Returns the fraction of surviving clones
        '''
        if self._pers is None:
            self._pers = self.nsur/self.non_missing
            self._pers[np.isnan(self._pers)] = 0            
            
        return self._pers
    
    @property 
    def dbas(self):
        '''
            Returns the difference in basal cells per clone
        '''
        return self.av_bas_cs[1:] - self.av_bas_cs[:-1]
    
    @property
    def dsup(self):
        '''
            Returns the difference in suprabasal cells per clone
        '''
        return self.av_sup_cs[1:] - self.av_sup_cs[:-1]

    @property
    def lcd(self):
        '''
            Returns the labelled cell density
        '''
        if self._lcd is None:
            self._lcd  = np.sum(self.bas_ls*(self.bas_ls >= 0),0)/self.non_missing
            
        return self._lcd
    
    
    @property
    def av_bas_cs(self):
        '''
            Returns the average number of basal cells per clone
        '''
        if self._av_bas_cs is None:
            if self.containsBasal:
                self._av_bas_cs = np.sum(self.bas_ls*(self.bas_ls > 0),0)/self.nsur
            else:
                self._av_bas_cs = np.sum(self.bas_ls*(self.bas_ls >= 0),0)/self.non_missing
                #self._av_bas_cs = np.mean(self.bas_ls,0)
            
        return self._av_bas_cs
    
    @property
    def max_bas_cs(self):
        '''
            Returns the maximum number of basal cells per clone
        '''
        return np.max(self.bas_ls,0)
    
    @property
    def max_tot_cs(self):
        '''
            Returns the maximum number of basal cells per clone
        '''
        return np.max(self.bas_ls,0)    
    
    @property
    def av_tot_cs(self):
        '''
            Returns the average number of total cells per clone
        '''
        if self._av_tot_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._av_tot_cs = np.sum(self.tot_ls*bas_mask*(self.tot_ls >= 0),0)/self.nsur
            else:
                self._av_tot_cs = np.sum(self.tot_ls*(self.tot_ls >= 0),0)/self.non_missing
                #self._av_tot_cs = np.mean(self.tot_ls,0)

        return self._av_tot_cs
    
    @property
    def av_sup_cs(self):
        '''
            Returns the average number of suprabasal cells per clone
        '''
        if self._av_sup_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0                
                self._av_sup_cs = np.sum(self.sup_ls*bas_mask*(self.sup_ls >= 0),0)/self.nsur
            else:
                self._av_sup_cs = np.sum(self.sup_ls*(self.sup_ls >= 0),0)/self.non_missing
                #self._av_sup_cs = np.mean(self.sup_ls,0)
            
        return self._av_sup_cs    
    
    @property
    def av_full_bas_cs(self):
        '''
            Returns average basal clone size for all clones    
        '''

        return self.lcd
    
    @property
    def av_full_tot_cs(self):
        '''
            Returns average total clone size for all clones    
        '''

        return np.sum(self.tot_ls*(self.tot_ls >= 0),0)/self.non_missing
    
    @property
    def av_full_sup_cs(self):
        '''
            Returns average suprabasal clone size for all clones    
        '''

        return np.sum(self.sup_ls*(self.sup_ls >= 0),0)/self.non_missing
       
    @property
    def se_bas_cs(self):
        '''
            Returns the standard error of basal cells per clone
        '''
        if self._se_bas_cs is None:
            if self.containsBasal:
                self._se_bas_cs = np.sqrt(np.sum((self.bas_ls > 0)*(self.bas_ls - self.av_bas_cs[None,:])**2,0))/self.nsur
            else:
                self._se_bas_cs = np.sqrt(np.sum((self.bas_ls >= 0)*(self.bas_ls - self.av_bas_cs[None,:])**2,0))/self.non_missing
            
        return self._se_bas_cs
    
    
    @property
    def se_tot_cs(self):
        '''
            Returns the standard error of total cells per clone
        '''
        if self._se_tot_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._se_tot_cs = np.sqrt(np.sum(bas_mask*(self.tot_ls - self.av_tot_cs[None,:])**2,0))/self.nsur
            else:
                self._se_tot_cs = np.sqrt(np.sum((self.tot_ls >= 0)*(self.tot_ls - self.av_tot_cs[None,:])**2,0))/self.non_missing
            
        return self._se_tot_cs
    
    @property
    def se_sup_cs(self):
        '''
            Returns the standard error of suprabasal cells per clone
        '''
        if self._se_sup_cs is None:
            if self.containsBasal:
                bas_mask = self.bas_ls > 0
                self._se_sup_cs = np.sqrt(np.sum(bas_mask*(self.sup_ls - self.av_sup_cs[None,:])**2,0))/self.nsur
            else:
                self._se_sup_cs = np.sqrt(np.sum((self.sup_ls >= 0)*(self.sup_ls - self.av_sup_cs[None,:])**2,0))/self.non_missing
            
        return self._se_sup_cs
    

    @property
    def se_full_bas_cs(self):
        '''
            Returns the standard error of basal cells per clone
        '''
        return self.se_lcd
    

    @property
    def se_full_tot_cs(self):
        '''
            Returns the standard error of total cells per clone
        '''
        return np.sqrt(np.sqrt(np.sum((self.tot_ls >= 0)*(self.tot_ls - self.av_full_tot_cs[None,:])**2,0))/self.non_missing)
    
    @property
    def se_full_sup_cs(self):
        '''
            Returns the standard error of suprabasal cells per clone
        '''
        return np.sqrt(np.sqrt(np.sum((self.sup_ls >= 0)*(self.sup_ls - self.av_full_sup_cs[None,:])**2,0))/self.non_missing)
    

    @property
    def se_pers(self):
        '''
            Returns the standard error of the clonal persistence
        '''

        if self._se_pers is None:
            self._se_pers = np.sqrt(self.pers*(1 - self.pers)/self.non_missing)
            self._se_pers[np.isnan(self._se_pers)] = 0

        return self._se_pers
    
    
    @property
    def se_lcd(self):
        ''' 
            Returns the standard error of the labelled cell density
        '''
        if self._se_lcd is None:
            self._se_lcd = np.sqrt(np.sum((self.bas_ls >= 0)*(self.bas_ls - self.lcd[None,:])**2,0))/self.non_missing
            
        return self._se_lcd


    @property 
    def ev_bas_cs(self):
        '''
            Returns the even bias for basal clone sizes
        '''
        if self._ev_bas_cs is None:
            self._ev_bas_cs = analysis.evenbias(self.bas_ls.T)
            
        return self._ev_bas_cs
    
    @property
    def ev_tot_cs(self):
        '''
            Returns the even bias for total clone sizes
        '''
        if self._ev_tot_cs is None:
            self._ev_tot_cs = analysis.evenbias(self.tot_ls.T)
            
        return self._ev_tot_cs
    
    @property
    def ev_sup_cs(self):
        '''
            Returns the even bias for suprabasal clone sizes
        '''
        if self._ev_sup_cs is None:
            self._ev_sup_cs = analysis.evenbias(self.sup_ls.T)
            
        return self._ev_sup_cs
    

    
    
    def bas_pdf(self, max_cs = None):
        '''
            Returns the pdf of basal cells per clone
        '''
        if self._bas_pdf is None or (max_cs is not None and self._bas_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._bas_pdf = analysis.basal_cont_pdf(self.bas_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._bas_pdf = analysis.joint_pdf(self.bas_ls, max_cs)
            
        return self._bas_pdf
    

    def full_bas_pdf(self, max_cs = None):
        '''
            Returns the pdf of basal cells per clone
        '''
        if self._full_bas_pdf is None or (max_cs is not None and self._full_bas_pdf.shape[1] != max_cs):
            self._full_bas_pdf = analysis.joint_pdf(self.bas_ls, max_cs)
            
        return self._full_bas_pdf
    
    def tot_pdf(self, max_cs = None):
        '''
            Returns the pdf of total cells per clone
        '''
        if self._tot_pdf is None or (max_cs is not None and self._tot_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._tot_pdf = analysis.basal_cont_pdf(self.tot_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._tot_pdf = analysis.joint_pdf(self.tot_ls, max_cs)
            
        return self._tot_pdf
    
    def full_tot_pdf(self, max_cs = None):
        '''
            Returns the pdf of total cells per clone
        '''
        if self._full_tot_pdf is None or (max_cs is not None and self._full_tot_pdf.shape[1] != max_cs):
            self._full_tot_pdf = analysis.joint_pdf(self.tot_ls, max_cs)
            
        return self._full_tot_pdf
    

    def sup_pdf(self, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone
        '''
        if self._sup_pdf is None or (max_cs is not None and self._sup_pdf.shape[1] != max_cs):
            if self.containsBasal:
                self._sup_pdf = analysis.basal_cont_pdf(self.sup_ls, self.bas_ls, self.nsur, max_cs)
            else:
                self._sup_pdf = analysis.joint_pdf(self.sup_ls, max_cs)
            
        return self._sup_pdf
    
    def full_sup_pdf(self, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone
        '''
        if self._full_sup_pdf is None or (max_cs is not None and self._full_sup_pdf.shape[1] != max_cs):
            self._full_sup_pdf = analysis.joint_pdf(self.sup_ls, max_cs)
            
        return self._full_sup_pdf
        

    def bas_cdf(self, max_cs = None):
        '''
            Returns the cdf of basal cells per clone
        '''
        return getCdf(self.bas_pdf(max_cs))

    def sup_cdf(self, max_cs = None):
        '''
            Returns the cdf of basal cells per clone
        '''
        return getCdf(self.sup_pdf(max_cs))
    
    def tot_cdf(self, max_cs = None):
        '''
            Returns the cdf of total cells per clone
        '''
        return getCdf(self.tot_pdf(max_cs))

        
    def surviving_at_time(self, t):
        '''
            Returns a mask of which clones are surviving at a given time
        '''
        return self.bas_ls[:,t] > 0        
        
    
    def div_first(self, subtimes = None):
        '''
            Clones that divided first given sub-times
                Use as experimental times to have the 
                right value of comparison
        '''
        if subtimes is None:
            subtimes = slice(None, None, None)
        
        return np.max(self.tot_ls[:, subtimes], 1) > 1
    
    
    def del_first(self, subtimes = None):
        '''
            Clones that delaminated first given sub-times
                Use as experimental times to have the 
        '''
        if subtimes is None:
            subtimes = slice(None, None, None)
        
        return np.max(self.tot_ls[:, subtimes], 1) == 1


    def fsur_subpop(self, subpop):
        '''
            Specify fraction of sub-population of clones that survive
                e.g. Fraction of surviving clones that divided first
        '''     
        
        if np.max(self.missing) == 0:
            return np.sum(self.bas_ls[subpop, :] > 0,0)/np.sum(subpop)
        else:
            return np.sum(self.bas_ls[subpop, :] > 0,0)/np.sum(self.bas_ls[subpop, :] >= 0,0)
   
        
    def wdiv_av_bas_cs(self, divbias = 0.5, subtimes = None):
        '''
            Returns the average number of basal cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first
                
            Use as experimental times to have the 
                right value of comparison
        '''    

        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)
        
        assert(divbias >= 0 and divbias <= 1)
        assert(np.max(self.missing) == 0)
        ''' 
            Currently missing data is not supported for this function
            
            Note to future self, if we have missing data, we need to do something like
                np.sum(self.bas_ls[div1,:],0)/np.sum(self.bas_ls[div1,:] >= 0, 0)
                
            But I need to test properly before implementation
        '''
        
        av_bas_div = np.sum(self.bas_ls[div1,:],0)/np.sum(div1)
        av_bas_del = np.sum(self.bas_ls[del1,:],0)/np.sum(del1)        
        
        if self.containsBasal:
            return (divbias*av_bas_div + (1 - divbias)*av_bas_del)/(divbias*self.fsur_subpop(div1) + (1 - divbias)*self.fsur_subpop(del1))

        else:
            return divbias*av_bas_div + (1-divbias)*av_bas_del
       
        
    def wdiv_av_sup_cs(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the average number of suprabasal cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
        
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)
        assert(divbias >= 0 and divbias <= 1)  
        assert(np.max(self.missing) == 0)
        ''' 
            Currently missing data is not supported for this function
            
            Note to future self, if we have missing data, we need to do something like
                np.sum(self.bas_ls[div1,:],0)/np.sum(self.bas_ls[div1,:] >= 0, 0)
                
            But I need to test properly before implementation
        '''        
        
        if self.containsBasal:
            bas_mask = self.bas_ls > 0   
            av_sup_div = np.sum(self.sup_ls[div1,:]*bas_mask[div1,:],0)/np.sum(div1)
            av_sup_del = np.sum(self.sup_ls[del1,:]*bas_mask[del1,:],0)/np.sum(del1)     
            return (divbias*av_sup_div + (1 - divbias)*av_sup_del)/(divbias*self.fsur_subpop(div1) + (1 - divbias)*self.fsur_subpop(del1))
            
        else:
            av_sup_div = np.sum(self.sup_ls[div1,:],0)/np.sum(div1)
            av_sup_del = np.sum(self.sup_ls[del1,:],0)/np.sum(del1)                
            return divbias*av_sup_div + (1-divbias)*av_sup_del
        
        
    def wdiv_av_tot_cs(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the average number of total cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
    
        assert(divbias >= 0 and divbias <= 1)
        assert(np.max(self.missing) == 0)
        ''' 
            Currently missing data is not supported for this function
            
            Note to future self, if we have missing data, we need to do something like
                np.sum(self.bas_ls[div1,:],0)/np.sum(self.bas_ls[div1,:] >= 0, 0)
                
            But I need to test properly before implementation
        '''
                
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)        
        av_tot_div = np.sum(self.tot_ls[div1,:],0)/np.sum(div1)
        av_tot_del = np.sum(self.tot_ls[del1,:],0)/np.sum(del1)        
        
        if self.containsBasal:
            bas_mask = self.bas_ls > 0   
            av_tot_div = np.sum(self.tot_ls[div1,:]*bas_mask[div1,:],0)/np.sum(div1)
            av_tot_del = np.sum(self.tot_ls[del1,:]*bas_mask[del1,:],0)/np.sum(del1)    
            return (divbias*av_tot_div + (1 - divbias)*av_tot_del)/(divbias*self.fsur_subpop(div1) + (1 - divbias)*self.fsur_subpop(del1))
            
        else:
            av_tot_div = np.sum(self.tot_ls[div1,:],0)/np.sum(div1)
            av_tot_del = np.sum(self.tot_ls[del1,:],0)/np.sum(del1)                    
            return divbias*av_tot_div + (1-divbias)*av_tot_del
        
                
    def wdiv_pers(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the fraction of surviving clones, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
    
        assert(divbias >= 0 and divbias <= 1)
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)            
        return divbias*self.fsur_subpop(div1) + (1-divbias)*self.fsur_subpop(del1)
    
    
    def wdiv_bas_pdf(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the pdf of basal cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
        
        assert(divbias >= 0 and divbias <= 1)
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)        
        max_cs = self.bas_ls.max()     
        
        if self.containsBasal:
            div_pdf = analysis.basal_cont_pdf(self.bas_ls[div1,:], self.bas_ls[div1,:], None, max_cs)
            del_pdf = analysis.basal_cont_pdf(self.bas_ls[del1,:], self.bas_ls[del1,:], None, max_cs)
            return (divbias*div_pdf*self.fsur_subpop(div1)[:,None] + (1-divbias)*del_pdf*self.fsur_subpop(del1)[:,None])/(divbias*self.fsur_subpop(div1)[:,None] + (1-divbias)*self.fsur_subpop(del1)[:,None])
        else:
            div_pdf = analysis.joint_pdf(self.bas_ls[div1,:], max_cs)
            del_pdf = analysis.joint_pdf(self.bas_ls[del1,:], max_cs)
            
            return divbias*div_pdf + (1-divbias)*del_pdf
      
        
    def wdiv_tot_pdf(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the pdf of total cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
        
        assert(divbias >= 0 and divbias <= 1)
        
        max_cs = self.tot_ls.max()
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)
                
        if self.containsBasal:
            div_pdf = analysis.basal_cont_pdf(self.tot_ls[div1,:], self.bas_ls[div1,:], None, max_cs)
            del_pdf = analysis.basal_cont_pdf(self.tot_ls[div1,:], self.bas_ls[div1,:], None, max_cs)
            return (divbias*div_pdf*self.fsur_subpop(div1)[:,None] + (1-divbias)*del_pdf*self.fsur_subpop(del1)[:,None])/(divbias*self.fsur_subpop(div1)[:,None] + (1-divbias)*self.fsur_subpop(del1)[:,None])
        else:
            div_pdf = analysis.joint_pdf(self.tot_ls[div1,:], max_cs)
            del_pdf = analysis.joint_pdf(self.tot_ls[div1,:], max_cs)
            return divbias*div_pdf + (1-divbias)*del_pdf
        
        
    def wdiv_sup_pdf(self, divbias = 0.5, subtimes = None):
        ''' 
            Returns the pdf of suprabasal cells per clone, 
                weighting by a bias to the sub-population of cells 
                which divided first        
        '''
        
        assert(divbias >= 0 and divbias <= 1)
        
        max_cs = self.sup_ls.max()  
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)          
        
        if self.containsBasal:
            div_pdf = analysis.basal_cont_pdf(self.sup_ls[div1,:], self.bas_ls[div1,:], None, max_cs)
            del_pdf = analysis.basal_cont_pdf(self.sup_ls[del1,:], self.bas_ls[del1,:], None, max_cs)
            return (divbias*div_pdf*self.fsur_subpop(div1)[:,None] + (1-divbias)*del_pdf*self.fsur_subpop(del1)[:,None])/(divbias*self.fsur_subpop(div1)[:,None] + (1-divbias)*self.fsur_subpop(del1)[:,None])
        else:
            div_pdf = analysis.joint_pdf(self.sup_ls[div1,:], max_cs)
            del_pdf = analysis.joint_pdf(self.sup_ls[del1,:], max_cs)
            
            return divbias*div_pdf + (1-divbias)*del_pdf        
        

    def fate_imbalance(self, exclude_missing_transitions = False, divbias = None, subtimes = None):
        ''' 
            Calculates the basal fate imbalance as defined to changes
                to the basal labelled cell density 
                
                Defined as Delta in 
                    Exp(Delta*(t2-t1)) = LCD(t2)/LCD(t1)
                    
            Flags:
                exclude_missing_transitions, set to True to exclude clones that have missing
                    data at that time-point or adjacent
        '''
        
        div1 = self.div_first(subtimes)
        del1 = self.del_first(subtimes)
                
        if divbias is None:
            return fate_imbalance(self.bas_ls, self.lcd, self.tvals, exclude_missing_transitions)
        else:
            assert(divbias >= 0 and divbias <= 1)
            div_imb = fate_imbalance(self.bas_ls[div1,:], None, self.tvals, exclude_missing_transitions)
            del_imb = fate_imbalance(self.bas_ls[del1,:], None, self.tvals, exclude_missing_transitions)            
            return (div_imb*divbias*self.fsur_subpop(div1)[:-1] + del_imb*(1 - divbias)*self.fsur_subpop(del1)[:-1])/(divbias*self.fsur_subpop(div1)[:-1] + (1 - divbias)*self.fsur_subpop(del1)[:-1])
    
    
    
    def allmoves(self, livesize = None, max_bas = None, max_sup = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between all states
            
            Only includes transitions from surviving clones            
            
            Stores in a multi-dimensional array of the form
            allmoves[time, bas_t-1, sup_t-1, bast, supt]
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis
        '''
        
        if livesize is None:
            livesize = self.live_bas_sb
            ntimes = self.ntimes - 1
        else:
            ntimes = livesize.shape[1] - 1
        
        if max_bas is None:
            max_bas = np.max(livesize[:,:,0]) + 2
            
        if max_sup is None:
            max_sup = np.max(livesize[:,:,1]) + 2
        
        matshape = (ntimes, max_bas, max_sup, max_bas, max_sup)
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
        
        if use_sparse:
            _allmoves = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves = np.zeros(matshape, dtype = 'int32')
        
        for c in range(livesize.shape[0]):
            for t in range(ntimes):
                bas0 = livesize[c,t,0]
                sup0 = livesize[c,t,1]
                bas1 = livesize[c,t+1,0]
                sup1 = livesize[c,t+1,1]
                
                if bas0 >= 0 and sup0 >= 0 and bas1 >= 0 and sup1 >= 0 and bas0 < max_bas and sup0 < max_sup and bas1 < max_bas and sup1 < max_sup:
                    _allmoves[t, bas0, sup0, bas1, sup1] += 1
        
        if use_sparse:
            _allmoves = _allmoves.to_coo()
        
        return _allmoves
            
        
    def allmoves_bas(self, livesize = None, max_bas = None, _allmoves = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between basal states
                
            Only includes transitions from surviving clones
            
            Stores in a multi-dimensional array of the form
            allmoves[time, bas_t-1, bast]
            
            This returns allmoves if self.ctypes is not specified with a warning
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis            
        '''

        if _allmoves is not None:
            return np.sum(_allmoves, axis = (2,4))
            
        if livesize is None:
            livesize = self.live_bas_sb
            ntimes = self.ntimes - 1
        else:
            ntimes = livesize.shape[1] - 1
        
        if max_bas is None:
            max_bas = np.max(livesize[:,:,0]) + 2
            
        matshape = (ntimes, max_bas, max_bas)
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
        
        if use_sparse:
            _allmoves_bas = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves_bas = np.zeros(matshape, dtype = 'int32')            
            
        for c in range(livesize.shape[0]):
            for t in range(self.ntimes - 1):
                bas0 = livesize[c,t,0]
                bas1 = livesize[c,t+1,0]
                if bas0 >= 0 and bas1 >= 0 and bas0 < max_bas and bas1 < max_bas:
                    _allmoves_bas[t, bas0, bas1] += 1
        
        if use_sparse:
            _allmoves_bas = _allmoves_bas.to_coo()        
        
        return _allmoves_bas
    
    
    def allmoves_tot(self, max_tot = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between total states
                
            Only includes transitions from clones with > 0 cells
            
            Stores in a multi-dimensional array of the form
            allmoves[time, tot_t-1, tott]
            
            This returns allmoves if self.ctypes is not specified with a warning
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis            
        '''
        
        if max_tot is None:
            max_tot = np.max(self.tot_ls) + 2
            
        matshape = (self.ntimes - 1, max_tot, max_tot)
        
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
            
        if use_sparse:
            _allmoves_tot = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves_tot = np.zeros(matshape, dtype = 'int32')
            
        for c in range(self.nclones):
            for t in range(self.ntimes - 1):
                tot0 = self.tot_ls[c,t]
                tot1 = self.tot_ls[c,t+1]
                if tot0 > 0 and tot1 >= 0 and tot0 < max_tot and tot1 < max_tot:
                    _allmoves_tot[t, tot0, tot1] += 1
        
        if use_sparse:
            _allmoves_tot = _allmoves_tot.to_coo()
            
        return _allmoves_tot    
    
    
    def allmoves_totwsheds(self, max_tot = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between total states inc shed
                
            Only includes transitions from clones with > 0 cells
                        
            Stores in a multi-dimensional array of the form
            allmoves[time, tot_t-1, tott]
            
            This returns allmoves if self.ctypes is not specified with a warning
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis            
        '''
        
        if max_tot is None:
            max_tot = np.max(self.tot_w_sheds) + 2
            
        matshape = (self.ntimes - 1, max_tot, max_tot)
        
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
            
        if use_sparse:
            _allmoves_tot = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves_tot = np.zeros(matshape, dtype = 'int32')
            
        for c in range(self.nclones):
            for t in range(self.ntimes - 1):
                tot0 = self.tot_w_sheds[c,t]
                tot1 = self.tot_w_sheds[c,t+1]
                if tot0 > 0 and tot1 >= 0 and tot0 < max_tot and tot1 < max_tot:
                    _allmoves_tot[t, tot0, tot1] += 1
        
        if use_sparse:
            _allmoves_tot = _allmoves_tot.to_coo()
            
        return _allmoves_tot
        
        
        
    def allmoves_by_type(self, livesize = None, max_bas = None, max_sup = None, 
                         ctypes = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between all states
            
            Only includes transitions from surviving clones            
            
            Stores in a multi-dimensional array of the form
            allmoves[time, bas_t-1, sup_t-1, bast, supt, clone type]
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis            
            
        ''' 
        
        if ctypes is None:
            ctypes = self.celltypes
        
        if len(ctypes) == 0:
            print('Warning: cell types not specified, returning allmoves')
            return self.allmoves(livesize, max_bas, max_sup)      
        
        if livesize is None:
            livesize = self.live_bas_sb
            ntimes = self.ntimes - 1
        else:
            ntimes = livesize.shape[1] - 1     
            
        if max_bas is None:
            max_bas = np.max(livesize[:,:,0]) + 2
            
        if max_sup is None:
            max_sup = np.max(livesize[:,:,1]) + 2              
                
        matshape = (ntimes, max_bas, max_sup, max_bas, max_sup, len(ctypes))
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
        
        if use_sparse:
            _allmoves = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves = np.zeros(matshape, dtype = 'int32')          
        

        for ct in range(len(ctypes)):
            cids = self.ctype_ids(ctypes[ct], livesize)
            for c in range(len(cids)):
                for t in range(ntimes):
                    bas0 = livesize[cids[c],t,0]
                    sup0 = livesize[cids[c],t,1]
                    bas1 = livesize[cids[c],t+1,0]
                    sup1 = livesize[cids[c],t+1,1]
                    if bas0 > 0 and sup0 >= 0 and bas1 >= 0 and sup1 >= 0 and bas0 < max_bas and sup0 < max_sup and bas1 < max_bas and sup1 < max_sup:
                        _allmoves[t, bas0, sup0, bas1, sup1, ct] += 1
        
        if use_sparse:
            _allmoves = _allmoves.to_coo()           
        
        return _allmoves
        
    
    
    def allmoves_bas_by_type(self, livesize = None, max_bas = None, 
                             ctypes = None, _allmoves_ctype = None, use_sparse = False):
        ''' 
            Calculates counts of transitions between all states
            
            Only includes transitions from surviving clones            
            
            Stores in a multi-dimensional array of the form
            allmoves[time, bas_t-1, bast, clone type]
            
            In some situations the problem becomes too memory intensive, 
                so we can use a sparse representation
            Current implementation of sparse matrices is slow. Should follow the prescription
                I used for the joint pdf in analysis            
            
        '''
        
        if _allmoves_ctype is not None:
            return np.sum(_allmoves_ctype, axis = (2,4))               
        
        if ctypes is None:
            ctypes = self.celltypes
        
        if len(ctypes) == 0:
            print('Warning: cell types not specified, returning allmoves')
            return self.allmoves(livesize, max_bas)      
        
        if livesize is None:
            livesize = self.live_bas_sb
            ntimes = self.ntimes - 1
        else:
            ntimes = livesize.shape[1] - 1     
            
        if max_bas is None:
            max_bas = np.max(livesize[:,:,0]) + 2

        matshape = (ntimes, max_bas, max_bas, len(ctypes))
        if np.log10(np.prod(matshape)) > 7.5:
            use_sparse = True
            print('Using sparse representation')
        
        if use_sparse:
            _allmoves_bas = sparse.DOK(matshape, dtype = np.int32, fill_value = 0)
        else:
            _allmoves_bas = np.zeros(matshape, dtype = 'int32')    

        for ct in range(len(ctypes)):
            cids = self.ctype_ids(ctypes[ct], livesize)
            for c in range(len(cids)):
                for t in range(ntimes):
                    bas0 = livesize[cids[c],t,0]
                    bas1 = livesize[cids[c],t+1,0]
                    if bas0 > 0 and bas1 >= 0 and bas0 < max_bas and bas1 < max_bas:
                        _allmoves_bas[t, bas0, bas1] += 1
        
        if use_sparse:
            _allmoves_bas = _allmoves_bas.to_coo()             
        
        return _allmoves_bas
    
    
    def sample(self, livesize = None, sample_size = None, update = True):
        '''
            Take a random sample with replacement from clones
        '''
        
        if livesize is None:
            livesize = self.livesize
            
        if sample_size is None:
            sample_size = self.nclones            
            
        sample = np.random.randint(0, livesize.shape[0], size = sample_size)
        
        if update:
            self.livesize = livesize[sample,:,:]
            self.reset_properties()        
                
        return livesize[sample,:,:]

    
    def boot(self, quantity, nboots = 1000, calc = 'se', seed = 0, sample_size = None):
        ''' 
            Can boostrap the standard error in any function from clones,
                does this in a slow way that minimises code
        '''
        
        assert(hasattr(self, quantity))
        
        bclones = self.copy()
        bdata = []
        np.random.seed(seed)

        bclones.reset_properties()

        for i in range(nboots):
            bclones.sample(self.livesize, sample_size)
            bdata.append(getattr(bclones, quantity))

        bdata = np.array(bdata)
        
        if calc == 'se':
            return np.std(bdata,0)
        elif calc == 'mean':
            return np.mean(bdata,0)
        elif calc == 'inv_se':
            return np.std(1/bdata,0)
        else:
            return bdata
        

    def bootpdf(self, quantity, nboots = 1000, calc = 'se', seed = 0, sample_size = None, kwargs = None):
        ''' 
            Create a boostrap of a pdf, assumes max_cs is a parameter
        '''

        assert(hasattr(self, quantity))
        
        if kwargs is None:
            kwargs = {}

        bclones = self.copy()
        bdata = []
        np.random.seed(seed)

        bclones.reset_properties()
        max_cs = getattr(self, quantity)().shape[-1] - 1

        getattr(self, quantity)().shape[-1]

        for i in range(nboots):
            bclones.sample(self.livesize, sample_size)
            bdata.append(getattr(bclones, quantity)(max_cs = max_cs, **kwargs))

        bdata = np.array(bdata)
        
        if calc == 'se':
            return np.std(bdata,0)
        elif calc == 'mean':
            return np.mean(bdata,0)
        else:
            return bdata

    
    
    def copy(self):
        return copy.deepcopy(self)
    

    def select_sublist(self, sublist, ca = None):
        '''
            Select a sub-population of clones from a given set of clones
                given as a boolean numpy array. This ensures that nmarked is handled correctly
        '''

        if ca is None:
            ca = self

        assert(len(sublist) == ca.nclones)

        if not hasattr(ca, 'nmarked'):
            ca.nmarked = [np.sum(sublist)]

        if ca.npops > 1: 
            if len(ca.nmarked) > 1:  
                for p in range(ca.npops):
                    ca.nmarked[p] = np.sum(sublist[ca.splits[p]:ca.splits[p+1]])
                    #ca.nmarked[p] = np.sum((sublist < ca.splits[p+1]) & (sublist >= ca.splits[p]))
            else:
                ca.nmarked[0] = np.sum(sublist)
        
        ca.livesize = ca.livesize[sublist,...]
        ca.reset_properties()
        return ca

        
    
    
    def sur_t0(self, t0 = 0, use_copy = True):
        '''
            Extract clones that are surviving at some zero time-point
        '''
        if use_copy:
            ca = self.copy()
        else:
            ca = self
        
        ca.select_sublist(ca.bas_ls[:,t0] > 0)
    
        return ca
    
        
    def max_tot(self, tvals = None, w_sheds = True):
        '''
            Calculate the maximum total number of cells in a clone accounting for 
                shed cells
        '''

        if tvals is None:
            if self._max_tot is None:
                if w_sheds:
                    self._max_tot = np.max(self.tot_w_sheds, 1)
                else:
                    self._max_tot = np.max(self.tot_ls, 1)
            return self._max_tot
        else:
            if w_sheds:
                return np.max(self.tot_w_sheds[:,tvals], 1)
            else:
                return np.max(self.tot_ls[:,tvals], 1)
            

    def subpopulation(self, subpop, use_copy = False):
        '''
            Extract a sub-population of clones
        '''

        if subpop is None:
            return self
        elif type(subpop) is FunctionType:
            return self.extract_pop(subpop, use_copy)
        elif isinstance(subpop, int):
            return self.cellpop(subpop, use_copy)
        elif getattr(self, subpop) is not None:
            return getattr(self, subpop)(use_copy = use_copy)
        else:
            print('Subpopulation not recognised')
            assert(0)


    def cellpop(self, p, use_copy = True):
        '''
            Returns a clonal population
        '''

        assert(p >= 0 and p < 1)

        if use_copy:
            ca = self.copy()
        else:
            ca = self

        ca.reset_properties()

        print("Warning: cellpop does nothing for experimental data")
        return ca 

        
    def extract_pop(self, popfn, use_copy = True):
        '''
            Extract a sub-population of clones for a specified function
        '''
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        popc = popfn(ca.livesize)
        ca.select_sublist(popc)        

        return ca

    
    def S4like(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit S-like behaviour (expand to 4+ cells)
        '''
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Slc = (self.max_tot(tvals) >= 4)
        ca.select_sublist(Slc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)          
             
        return ca
        

    def Slike(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit S-like behaviour (expand to 3+ cells)
        '''
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Slc = (self.max_tot(tvals) >= 3)
        ca.select_sublist(Slc)
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)          
             
        return ca
    
    def nlike(self, n, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit S-like behaviour (expand to 3+ cells)
        '''
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Slc = (self.max_tot(tvals) == n)
        ca.select_sublist(Slc)
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)          
             
        return ca    
        
        
    def S2like(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit S2-like behaviour (expand to 2+ cells)
        '''  
         
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        S2lc = (self.max_tot(tvals) >= 2)
        ca.select_sublist(S2lc)
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)            
       
        return ca
    
    def TD3like(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit TA-like or D-like behaviour (expand to 3 or less cells)
        '''   
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Tlc = (self.max_tot(tvals) <= 3)
        ca.select_sublist(Tlc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)   
               
        return ca
        
    
    def Tlike(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit TA-like behaviour (expand to 2 cells)
        '''   
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Tlc = (self.max_tot(tvals) == 2)
        ca.select_sublist(Tlc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)   
               
        return ca
    

    def T3like(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit TA-like behaviour (expand to 3 cells)
        '''   
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Tlc = (self.max_tot(tvals) == 3)
        ca.select_sublist(Tlc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)   
            
        return ca
    
    def CPlike(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit TA-like behaviour (expand to 2-3 cells)
        '''   
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Tlc = (self.max_tot(tvals) == 2) | (self.max_tot(tvals) == 3)
        ca.select_sublist(Tlc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)   
            
        return ca        
    
    
    def Dlike(self, tvals = None, use_copy = True):
        '''
            Extract only clones that exhibit D-like behaviour (maximum one cell)
        '''        
        
        if use_copy:
            ca = self.copy()
        else:
            ca = self

        Dlc = (self.max_tot(tvals) == 1)
        ca.select_sublist(Dlc)        
        
        ca.reset_properties()        
        if self.use_t0:
            if tvals is None:
                t0 = 0
            else:
                t0 = tvals[0]
            ca = ca.sur_t0(t0, False)   
                   
        return ca       
    
    @property
    def trajectories(self):
        '''
            Returns list of trajectories
        '''
        
        traj = []


        for c in range(self.livesize.shape[0]):
            trajc = [np.array([1,0])]
            for t in range(self.livesize.shape[1]):
                if (self.livesize[c,t,:] != trajc[-1]).any():
                    trajc.append(self.livesize[c,t,:])
            traj.append(np.array(trajc))
            
        return traj
        
    
    @property
    def transitions(self):
        '''
            Returns list of transitions
        '''
        traj = self.trajectories
        
        trans = []
        
        for c in range(self.livesize.shape[0]):
            trans.append(traj[c][1:] - traj[c][:-1])
            
        return trans
    
    def tot_znz(self, verbose = False):
        '''
            Finds events where the total number of cells goes to zero and then to a non-zero number of cells
        '''
        if np.min(self.tot_ls) < 0:
            corr_tot_ls = miss_corrected(self.tot_ls.copy(), verbose)
        else:
            corr_tot_ls = self.tot_ls
        return (corr_tot_ls[:,:-1] == 0) & (corr_tot_ls[:,1:] > 0) & (self.livesize[:,:-1,0] >= 0)
        

    def bas_znz(self, verbose = False):
        ''' 
            Finds events where the basal number of cells goes to zero and then to a non-zero number of cells
        '''
        if np.min(self.bas_ls) < 0:
            corr_bas_ls = miss_corrected(self.bas_ls.copy(), verbose)
        else:
            corr_bas_ls = self.bas_ls
        return (corr_bas_ls[:,:-1] == 0) & (corr_bas_ls[:,1:] > 0) & (self.livesize[:,:-1,0] >= 0)

    @property
    def rel_sup_bas(self):
        '''
            Relative basal to suprabasal count in a clone
        '''
        #x = self.av_sup_cs/self.av_bas_cs
        x = (self.av_full_tot_cs - self.lcd)/self.lcd
        x[np.isnan(x)] = 0
        return x

    def repair_znz_miscounts(self):
        ''' 
            Replace entries where we go to either zero cells or 
                zero basal cells and then to a non-zero number of cells
        '''    
        
        lls = self.livesize[:,:-1,:]
        
        bas_mistakes = self.bas_znz()
        while np.sum(bas_mistakes) > 0:
            if self.verbose:
                print('Repairing basal CS mistakes for', np.where(np.sum(bas_mistakes,1) > 0)[0])
                print(np.transpose(self.livesize[np.where(np.sum(bas_mistakes,1) > 0)[0],:,:],(0,2,1)))
            lls[bas_mistakes,:] = -100
            bas_mistakes = self.bas_znz()

        tot_mistakes = self.tot_znz()
        while np.sum(tot_mistakes) > 0:
            if self.verbose:
                print('Repairing total CS mistakes for', np.where(np.sum(tot_mistakes,1) > 0)[0])
                print(np.transpose(self.livesize[np.where(np.sum(tot_mistakes,1) > 0)[0],:,:],(0,2,1)))
            lls[tot_mistakes,:] = -100
            tot_mistakes = self.tot_znz()
            
        self.reset_properties()
        return self    
         
         
    def repair_extinct_then_missing(self):
        ''' 
            Repair clones that first go extinct then are missing
        '''
        
        print('Warning: repair_extinct_then_missing function is not fully tested')
        print('Please test thoroughly on first use')
        
        for c in range(self.nclones):
            if np.sum(self.tot_ls[c,:] == 0) > 0:
                tidx = np.where(self.tot_ls[c,:] == 0)[0][-1]
                #Setting clones which are zero after a certain timepoint
                if np.sum(self.livesize[c,tidx:,:] > 0) == 0 and np.sum(self.livesize[c,tidx:,:] < 0) > 0: 
                    self.livesize[c,tidx:,:] = 0
        self.reset_properties()


class xanalysis(lanalysis):
    def __init__(self, livesize, popnames=None, tvals=None, containsBasal=True, suprabasal=True, **kwargs):
        '''
            Analysis of clones from experiments given a live-size is provided,
                handles missing data,
        '''
        
        lanalysis.__init__(self, livesize, popnames, tvals, containsBasal, suprabasal, **kwargs)
        
        


class sanalysis(lanalysis):
    def __init__(self, results = None, popnames = None, skip = 1, tvals = None, containsBasal = True, suprabasal = True, 
                 #If don't provide results can provide the following
                 cellids = None, ncells = None, ancs = None, nsbcells = None, livesize = None, t0 = 0,
                  exp_times = None, **kwargs):
        '''
            Analysis of clones from simulations
            
            Initially we calculate the live-size of each clone
            
            Handles population quantities which are not included in the base class,
                as they require more complex handling of missing data
            Assums that there is no missing data
            
            Flags:
                containsBasal, set to True to only include clones that contain basal cells
                suprabasal, set to True to include suprabasal cells in the analysis
                    Currently coded to only work with one supra-basal layer
        '''
        
        if results is not None:
            cellids = results['cellid']
            ncells = results['ncells'].copy()
            nsbcells = results['nsbcells']
            ancs = results['anc']
            
            if 'tiss' in results:
                sb_born = results['tiss'].cellpops[-1].born #extract suprabasal
            else:
                sb_born = None
         
        #Set times
        self.skip = skip
        
        if livesize is None:
            #Attributes
            self.npops = ncells.shape[1]
            self.inducttime = 0
            self.nmarked = ncells[self.inducttime]
            self.nsb0 = nsbcells[self.inducttime] #number of suprabasal cells at t =0
            self.splits = self.calc_splits()          
            self._nclones = np.sum(self.nmarked)

            if tvals is None:
                self.tvals = np.arange(self.inducttime, len(ncells), skip)
            else:
                self.tvals = np.array(tvals)
            tvals = self.tvals

            self._ntimes  = len(self.tvals)

            #For calculating clone-size, use pandas series because vectorisation is allowed
            self.clone_inverseid = pd.Series(dict(zip(cellids[self.inducttime], np.arange(self.nclones))), dtype = 'int32')
            self.ancs = {p: pd.Series(ancs[p], dtype = 'int32') for p in range(self.npops + 1)}
            
            #Storing of when suprabasal cells are born, allows re-running suprabasal layer
            if sb_born is not None:
                self.sb_born = pd.Series(sb_born, dtype = 'int32')
            else:
                self.sb_born = None
                        
            #Calculate the live-size of each clone, all other quantities are calculated on the fly
            livesize = self.calc_livesize(cellids, ncells, ancs, nsbcells)
        else:
            self.livesize = livesize
            self._ntimes = None
            self.nmarked = [livesize.shape[0]]


        #Defining experimental times
        self.exp_times = exp_times
        if self.exp_times is not None and len(self.exp_times) >= self.ntimes: 
            self.exp_times = None

        if self.exp_times is not None:
            t0 = exp_times[0]
            self.t0 = t0

        lanalysis.__init__(self, livesize, popnames, tvals, containsBasal, suprabasal, t0 = t0, copy_livesize = False, **kwargs)

        #Enforce no missing data
        if self.nclones > 0:
            assert(np.min(self.livesize) >= 0)
        
        
    def reset_properties(self):
        #Reset base quantities
        lanalysis.reset_properties(self)

        #Population quantities
        if np.sum(self.nmarked) != self.livesize.shape[0]:
            if self.npops > 1:
                print('Inconsistent number of clones marked by pop')
                assert(0)

            self.nmarked = [self.livesize.shape[0]]

        self.splits = self.calc_splits() #Splits between different population indices
        self._popsur = None #Survival probability of each population
        self._pop_ave_bas_cs = None #Average number of basal cells per clone labelled by a certain population
        self._pop_ave_tot_cs = None #Average number of total cells per clone labelled by a certain population        
        self._pop_ave_sup_cs = None #Average number of suprabasal cells per clone labelled by a certain population
        self._pop_bas_pdf = None #PDF of basal cells per clone labelled by a certain population
        self._pop_tot_pdf = None #PDF of total cells per clone labelled by a certain population
        self._pop_sup_pdf = None #PDF of suprabasal cells per clone labelled by a certain population
        self._ave_cs_by_pop = None # Average cells of type per clone
        self._pop_ave_cs_by_pop = None # Average cells of type per clone labelled by a certain population
               


    def calc_splits(self):
        '''
            Calculate splits
        '''
        return np.array([0] + list(np.cumsum(self.nmarked)) + [np.sum(self.nmarked) + self.nsb0])



    def max_tot(self, tvals = None, w_sheds = True):
        '''
            Calculate the maximum total number of cells in a clone accounting for 
                shed cells, using experimental time-points
        '''
        if tvals is None:
            if self._max_tot is None:
                if self.exp_times is None:
                    if w_sheds:
                        self._max_tot = np.max(self.tot_w_sheds, 1)
                    else:
                        self._max_tot = np.max(self.tot_ls, 1)
                else:
                    #print('Using experimental times for max_tot')
                    if w_sheds:
                        self._max_tot = np.max(self.tot_w_sheds[:,self.exp_times], 1)
                    else:
                        self._max_tot = np.max(self.tot_ls[:,self.exp_times], 1)
            return self._max_tot
        else:
            if w_sheds:
                return np.max(self.tot_w_sheds[:,tvals], 1)
            else:
                return np.max(self.tot_ls[:,tvals], 1)

    
    def xfrac(self, tvals = None, exp_times = False):
        ''' 
            Returns fraction of extinction clones, with option
                to use experimental times
        '''
        if exp_times:
            return lanalysis.xfrac(self, self.exp_times)
        else:
            return lanalysis.xfrac(self, tvals)

               
    def rerun_suprabasal(self, shed, refract = 0, sbas_cv = 1.0):
        '''
            Reruns the suprabasal layer and calculates new live-size
        '''
        
        if hasattr(self, 'sb_born') is False or hasattr(self, 'clone_inverseid') is False or hasattr(self, 'ancs') is False:
            print('No suprabasal layer to rerun, not implemented')
            assert(0)
        
        cloneidx = self.clone_inverseid[self.ancs[self.npops][self.sb_born.index.to_numpy()]].values        
        self.livesize = fast_rerun_supra(self.livesize, cloneidx, self.sb_born.values, shed, refract, self.skip, sbas_cv)
        
        self.reset_properties()

        assert(self.tot_znz().sum() == 0)
        assert(self.bas_znz().sum() == 0)
        
        return self.livesize
    
    @property
    def popsur(self):
        '''
            Returns the survival probability of each population
        '''
        if self._popsur is None:
            self._popsur = np.zeros((self.npops, self.ntimes)) # Survival probability of clones marked from specific populations
            for p in range(self.npops):
                self._popsur[p,:] = np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0,0)/np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],0])

        return self._popsur
    
    @property
    def pop_ave_bas_cs(self):
        '''
            Returns the average number of basal cells per clone labelled by a certain population
        '''
        if self._pop_ave_bas_cs is None:
            self._pop_ave_bas_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_ave_bas_cs[p,:] = np.sum(self.bas_ls[self.splits[p]:self.splits[p+1],:],0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_bas_cs[p,:] = np.mean(self.bas_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_bas_cs
    
    @property
    def pop_ave_tot_cs(self):
        '''
            Returns the average number of total cells per clone labelled by a certain population
        '''
        if self._pop_ave_tot_cs is None:
            self._pop_ave_tot_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0                    
                    self._pop_ave_tot_cs[p,:] = np.sum(self.tot_ls[self.splits[p]:self.splits[p+1],:]*bas_mask,0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_tot_cs[p,:] = np.mean(self.tot_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_tot_cs
    
    @property
    def pop_ave_sup_cs(self):
        '''
            Returns the average number of suprabasal cells per clone labelled by a certain population
        '''
        if self._pop_ave_sup_cs is None:
            self._pop_ave_sup_cs = np.zeros((self.npops, self.ntimes))
            
            for p in range(self.npops):
                if self.containsBasal:
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0                    
                    self._pop_ave_sup_cs[p,:] = np.sum(self.sup_ls[self.splits[p]:self.splits[p+1],:]*bas_mask,0)/self.popsur[p,:]/self.nmarked[p]
                else:
                    self._pop_ave_sup_cs[p,:] = np.mean(self.sup_ls[self.splits[p]:self.splits[p+1],:],0)

        return self._pop_ave_sup_cs
                    

    def pop_bas_pdf(self, max_cs = None):
        '''
            Returns the pdf of basal cells per clone labelled by a certain population
        '''
        if self._pop_bas_pdf is None or (max_cs is not None and (self._sup_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.bas_ls.max()
                
            self._pop_bas_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_bas_pdf[p,:,:] = analysis.basal_cont_pdf(self.bas_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_bas_pdf[p,:,:] = analysis.joint_pdf(self.bas_ls[self.splits[p]:self.splits[p+1],:], max_cs)

        return self._pop_bas_pdf
    

    def pop_tot_pdf(self, max_cs = None):
        '''
            Returns the pdf of total cells per clone labelled by a certain population
        '''
        if self._pop_tot_pdf is None or (max_cs is not None and (self._pop_tot_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.tot_ls.max()
                
            self._pop_tot_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_tot_pdf[p,:,:] = analysis.basal_cont_pdf(self.tot_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_tot_pdf[p,:,:] = analysis.joint_pdf(self.tot_ls[self.splits[p]:self.splits[p+1],:], max_cs)
                    
                    
        return self._pop_tot_pdf
    
    
    def pop_sup_pdf(self, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone labelled by a certain population
        '''
        if self._pop_sup_pdf is None or (max_cs is not None and (self._pop_sup_pdf.shape[1] != max_cs).any()):
            if max_cs is None:
                max_cs = self.sup_ls.max()
                
            self._pop_sup_pdf = np.zeros((self.npops, self.ntimes, max_cs + 1))
            
            for p in range(self.npops):
                if self.containsBasal:
                    self._pop_sup_pdf[p,:,:] = analysis.basal_cont_pdf(self.sup_ls[self.splits[p]:self.splits[p+1],:], self.bas_ls[self.splits[p]:self.splits[p+1],:], None, max_cs)
                else:
                    self._pop_sup_pdf[p,:,:] = analysis.joint_pdf(self.sup_ls[self.splits[p]:self.splits[p+1],:], max_cs)
                    
                    
        return self._pop_sup_pdf
    
    @property
    def ave_cs_by_pop(self):
        '''
            Average cells of type per clone
        '''
        
        if self._ave_cs_by_pop is None:
            if self.containsBasal:
                self._ave_cs_by_pop  = np.sum(self.livesize,0)/self.nsur[:,None]
            else:
                self._ave_cs_by_pop  = np.mean(self.livesize,0)
                
        return self._ave_cs_by_pop
    
    @property
    def pop_ave_cs_by_pop(self):
        '''
            Average cells of type per clone, given clones marking a specific population
        '''
        
        if self._pop_ave_cs_by_pop is None:
            if self.containsBasal:
                self._pop_ave_cs_by_pop  = np.zeros((self.npops, self.ntimes, self.npops + 1))
                for p in range(self.npops):
                    bas_mask = self.bas_ls[self.splits[p]:self.splits[p+1],:] > 0
                    self._pop_ave_cs_by_pop[p,:,:] = np.sum(self.livesize[self.splits[p]:self.splits[p+1],:,:]*bas_mask[:,:,None],0)/self.popsur[p,:,None]/self.nmarked[p]
            else:
                self._pop_ave_cs_by_pop  = np.mean(self.livesize[self.splits[p]:self.splits[p+1],:,:],0)
                
        return self._pop_ave_cs_by_pop
        

    def wpop_av_bas_cs(self, bias = None):
        '''
            Returns the average number of basal cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.av_bas_cs
                
        if self.containsBasal:
            return np.sum(self.pop_ave_bas_cs*bias[:,None]*self.popsur, 0)/np.sum(bias[:,None]*self.popsur,0)
        else:
            return np.sum(self.pop_ave_bas_cs*bias[:,None], 0)/np.sum(bias)
        
        
        
    def wpop_av_tot_cs(self, bias = None):
        '''
            Returns the average number of total cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.av_tot_cs
            
        if self.containsBasal:
            return np.sum(self.pop_ave_tot_cs*bias[:,None]*self.popsur, 0)/np.sum(bias[:,None]*self.popsur,0)
        else:
            return np.sum(self.pop_ave_tot_cs*bias[:,None], 0)/np.sum(bias)


    def wpop_av_sup_cs(self, bias = None):
        '''
            Returns the average number of suprabasal cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.av_sup_cs
        
        if self.containsBasal:
            return np.sum(self.pop_ave_sup_cs*bias[:,None]*self.popsur, 0)/np.sum(bias[:,None]*self.popsur,0)
        else:
            return np.sum(self.pop_ave_sup_cs*bias[:,None], 0)/np.sum(bias)


    def wpop_bas_pdf(self, bias = None, max_cs = None):
        '''
            Returns the pdf of basal cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.bas_pdf(max_cs)
                
        if self.containsBasal:
            return np.sum(self.pop_bas_pdf(max_cs)*bias[:,None,None]*self.popsur[:,:,None], 0)/np.sum(bias[:,None,None]*self.popsur[:,:,None],0)
        else:
            return np.sum(self.pop_bas_pdf(max_cs)*bias[:,None,None], 0)/np.sum(bias)
    
    
    def wpop_tot_pdf(self, bias = None, max_cs = None):
        '''
            Returns the pdf of total cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.tot_pdf(max_cs)
                
        if self.containsBasal:
            return np.sum(self.pop_tot_pdf(max_cs)*bias[:,None,None]*self.popsur[:,:,None], 0)/np.sum(bias[:,None,None]*self.popsur[:,:,None],0)
        else:
            return np.sum(self.pop_tot_pdf(max_cs)*bias[:,None,None], 0)/np.sum(bias)
        
        
    def wpop_sup_pdf(self, bias = None, max_cs = None):
        '''
            Returns the pdf of suprabasal cells per clone, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.sup_pdf(max_cs)
                
        if self.containsBasal:
            return np.sum(self.pop_sup_pdf(max_cs)*bias[:,None,None]*self.popsur[:,:,None], 0)/np.sum(bias[:,None,None]*self.popsur[:,:,None],0)
        else:
            return np.sum(self.pop_sup_pdf(max_cs)*bias[:,None,None], 0)/np.sum(bias)


    def wpop_pers(self, bias = None):
        '''
            Returns the fraction of surviving clones, 
                weighting by a set bias to particular populations
                e.g. if bias = [0.5, 0.5] then we weighted evenly between 2 populations
                bias need not be normalised
        '''
        
        if bias is None or len(bias) != self.npops:
            if len(bias) != self.npops:
                print('Incorrect bias length, using uniform weighting')
            return self.pers
                
        return np.sum(bias[:,None]*self.popsur, 0)/np.sum(bias)
            


    def calc_livesize(self, cellids = None, ncells = None, ancs = None, nsbcells = None, 
                      results = None, pops = None, inititialise = True):
        '''
            Calculate the live-size of each clone
        '''
        
        if inititialise:
            self.livesize = np.zeros((self.nclones, self.ntimes, self.npops + 1), dtype = 'int32')
        
        if results is not None:
            cellids = results['cellid']
            ncells = results['ncells']
            nsbcells = results['nsbcells']
            ancs = results['anc']
            
        if pops is None:
            pops = np.arange(self.npops + 1)
        elif np.ndim(pops) == 0:
            pops = [pops]
            
        if self.ancs is None:
            anc1 = {p: pd.Series(ancs[p], dtype = 'int32') for p in pops}
            self.ancs = anc1
        else:
            anc1 = self.ancs
        
        for t in range(len(cellids)):
            tt = int(self.tvals[t]/self.skip)
            assert tt*self.skip <= len(ncells) , 'Time out of range at ' + str(tt)
            assert len(cellids[int(tt)]) == np.sum(ncells[tt*self.skip]) + nsbcells[tt*self.skip] , 'Time out of range at ' + str(tt)
            self.livesize = analysis.update_live_sizes(self.livesize, self.npops, t, ncells[tt*self.skip], nsbcells[tt*self.skip], cellids[tt], anc1, self.clone_inverseid,
                                                       pops = pops)
        
        return self.livesize
        

    def recalculate_livesize(self, new_results):
        '''
            Recalculate livesize if we have new results
        '''
        
        self.livesize = self.calc_livesize(results = new_results, pops = None, inititialise = True)
        self.reset_properties()
        return self.livesize
        
        
    def recalc_supra_livesize(self, new_results):
        '''
            Re-calculate live-size for suprabasal cells only
        '''
        self.livesize[:,:,-1] = 0
        self.livesize = self.calc_livesize(results = new_results, pops = self.npops, inititialise = False)
        self.reset_properties()
        return self.livesize
    

    def cellpop(self, p, use_copy = True):
        '''
            Returns a clonal population
        '''

        assert(p >= 0 and p < self.npops + 1)

        if use_copy:
            ca = self.copy()
        else:
            ca = self

        mask = np.zeros(self.nclones, dtype = 'bool')
        mask[self.splits[p]:self.splits[p+1]] = True
        ca.select_sublist(mask)        

        return ca
    
    
def fate_imbalance(bas_ls, lcd = None, tvals = None, exclude_missing_transitions = False):
    ''' 
        Calculates the basal fate imbalance as defined to changes
            to the basal labelled cell density 
            
            Defined as Delta in 
                Exp(Delta*(t2-t1)) = LCD(t2)/LCD(t1)
                
        Flags:
            exclude_missing_transitions, set to True to exclude clones that have missing
                data at that time-point or adjacent
    '''
    
    if tvals is None:
        dt = 1
    else:
        dt = tvals[1:]-tvals[:-1]
    
    if exclude_missing_transitions:
        mask = (bas_ls[:,:-1] >= 0)*(bas_ls[:,1:] >= 0)
        lcd2 = np.sum(bas_ls[:,1:] *mask,0)
        lcd1 = np.sum(bas_ls[:,:-1]*mask,0)
        imbalance = np.log(lcd2/lcd1)/dt
        imbalance[lcd2 == 0.0] = 0
    else:
        if lcd is None:
            lcd = np.sum(bas_ls*(bas_ls >= 0), 0)
        imbalance = np.log(lcd[1:]/lcd[:-1])/dt
        imbalance[lcd[1:] == 0.0] = 0
    
    return imbalance



@njit
def fast_rerun_supra(livesize, cloneidx, born, shed, refract, skip, sbas_cv):
    ''' 
        Re-running the suprabasal layer with numbas

    '''
    
    n = len(cloneidx)
    assert(n == len(cloneidx))
    
    shed_times = lifetime(n, shed, sbas_cv) + refract
    livesize[:,:,-1] = 0

    for i in range(len(cloneidx)):
        livesize[cloneidx[i], int_ceil(born[i]/skip):next_floor((born[i] + shed_times[i])/skip),-1] += 1
        
    return livesize


@njit
def lifetime(ncells = 1, drate = None, cv_div = 1.0):
    '''
        Generate lifetime of ncells from population
        
    '''
    
    if drate < 1:
        return np.ones(ncells)
    elif cv_div == 1.0:
        return np.random.exponential(drate - 1, ncells) + 1
    else:
        return np.random.gamma(1/cv_div**2, (drate - 1)*cv_div**2, ncells) + 1
    
    
        
        
        
        
def label_calc(pop_ca, props, quantity, args=None, norm=False, sur=False):
    '''
        Calculate a quantity given a specified labelling bias between populations
        
        Input:
            - clone-analysis objects by populations as a list
            - props is the proportion of the ensemble that is labelled by each population
            - quantity is the quantity/function to calculate
            - args is a dictionary of arguments to pass if quantity is a function
                Pass in {} to evaluate at defaults
        
    '''
    
    assert(len(pop_ca) == len(props))
    
    if norm:
        props = np.array(props) / np.sum(props)
    
    props = np.array(props)
    q = 0
    pers_weighted_sum = np.zeros(len(getattr(pop_ca[0], quantity)))

    for i in range(len(pop_ca)):
        x = getattr(pop_ca[i], quantity).copy()
        
        if args is not None:
            x = x(**args)
        x[np.isnan(x)] = 0

        if sur:
            pers_weighted_sum += props[i] * pop_ca[i].pers  # accumulate weighted pers
            q += props[i] * pop_ca[i].pers * x              # weight by both props and pers
        else:
            q += props[i] * x
    
    if sur:
        # Normalize by total weighted survival
        nonzero = pers_weighted_sum > 0
        q[nonzero] /= pers_weighted_sum[nonzero]
        q[~nonzero] = 0
            
    return q

#boot(self, quantity, nboots = 1000, calc = 'se', seed = 0, sample_size = None)
def boot_label_calc(pop_ca, props, quantity, nboots = 1000, calc = 'se', seed = 0, 
                    sample_size = None, norm = False, sur = False):
    '''
        Bootstrap a quantity given a specified labelling bias between populations
        
        Use sample_size[i]
    '''

    assert(len(pop_ca) == len(props))
    
    if norm:
        props = np.array(props)/np.sum(props)

    props = np.array(props)
    q = 0
    pers_weighted_sum = np.zeros(len(getattr(pop_ca[0], quantity)))

    for i in range(len(pop_ca)):
        ss = int(sample_size * props[i]) if sample_size is not None else None
        x = pop_ca[i].boot(quantity, nboots, None, seed, ss)
        x[np.isnan(x)] = 0

        if sur:
            pers_weighted_sum += props[i] * pop_ca[i].pers
            x *= pop_ca[i].pers
            
        q += props[i] * x
            
    if sur:
        # Normalize by total weighted survival
        nonzero = pers_weighted_sum > 0
        q[:, nonzero] /= pers_weighted_sum[nonzero]
        q[:, ~nonzero] = 0

    if calc == 'se':
        return np.std(q, 0)
    elif calc == 'mean':
        return np.mean(q, 0)
    else:
        return q
    
def miss_corrected_2d(ls, verbose = False):
    ''' 
        Replace entries with the next time entries until no errors are left
            Missing entries are defined as negative values
    '''
    
    lls = ls[:,:-1]
    rls = ls[:,1:]
    
    rmiss = (rls >= 0) & (lls < 0)
    if np.sum(rmiss) > 0:
        if verbose:
            print('Repairing', np.where(np.sum(rmiss,1) > 0)[0])
            print('Before:', ls[np.sum(rmiss,1) > 0,:])
        lls[rmiss] = rls[rmiss]
        if verbose:
            print('After:', ls[np.sum(rmiss,1) > 0,:])    
    else:
        if verbose:
            print('No errors left')
    
    rmiss = (rls >= 0) & (lls < 0)
    if np.sum(rmiss) > 0:
        if verbose:
            print('Fixing recursively')
        ls = miss_corrected_2d(ls, verbose)

    return ls
        
def miss_corrected(ls, verbose = False):
    ''' 
        Replace entries with the next time entries until no errors are left
            Missing entries are defined as negative values
    '''
    
    if np.ndim(ls) == 2:
        return miss_corrected_2d(ls, verbose)
    else:
        for i in range(np.shape(ls)[2]):
            if verbose:
                print('Axis', i)
            ls[:,:,i] = miss_corrected_2d(ls[:,:,i], verbose)
        return ls
    

def getCdf(pdf):
    '''
        Converts pdf to cdf, works provided the values are kept along last dimension,
        Assumes but does not check normalisation
    '''
    return 1 - np.cumsum(pdf, axis = -1)
    #return np.cumsum(pdf[...,::-1], axis = -1)[...,::-1]   

@njit
def next_floor(x):
    '''
        Used to specify the next integer
    '''
    return int(1 + np.floor(x))

@njit
def int_ceil(x):
    '''
        Used to specify the next integer
    '''
    return int(np.ceil(x))