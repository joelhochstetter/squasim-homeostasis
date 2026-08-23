import numpy as np
from utils import *
import copy
from sortedcontainers import SortedSet
import pandas as pd
import scipy
from scipy.stats.sampling import NumericalInversePolynomial


# cell population class

class cellulation():
    '''
        Cell population class:
            specifies population of cells who have same rules for division vs differentiation
            
            in this version of the code only exponential division with fixed rate and prob is implemented

            To implement custom division rules, specify cusdivprobs = [] as a list and and cusdivDaughters = [] as a list of daughter types,
                where each element should be a pair of daughters

    '''
    def __init__(self, points, divtime = 1000, divprob = 0.5, totpts = 0, thetas = None, dx = 0.01, 
                 assprob = 0.0, difprob = 0.0, terprob = 0.0, daughterType = -1, 
                 fixchild = False, popidx = None, refract = 0, 
                 verbose = False, 
                 corrSym = 0, corrAss = 0,
                 popname = None, dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, 
                 cusdivprobs = 0, cusdivDaughters = 0,
                 daughter_tau = 0, cv_div = 1.0, **extras):
    
        if popname is None:
            popname = str(popidx)
        self.popname = popname

        self.poptype = 'stochastic'

        self.ncells  = len(points) #current number of cells
        
        self.popidx = popidx
        
        self.divtime  = divtime  # division time
        self.divprob  = divprob  #divprob*np.ones(self.ncells), probability to symmetrically divide
        self.refract  = refract #refractory time
        self.cv_div   = cv_div #CV of division times, 1.0 gives exponential distribution, < 1: gives Gamma distribution

        self.assprob = assprob #probability to assymemtrically divide
        self.difprob = difprob #probability to differentiate
        self.terprob = terprob #probability to terminally divide
        assert(self.assprob >= 0 and self.difprob >= 0 and self.terprob >= 0 and self.divprob >= 0)
        
        self.corrSym = corrSym #cell decision correlation for stochastic decisions for sisters born into this population
        self.corrAss = corrAss #cell decision correlation  for stochastic decisions resulting from asymmetric division from this population
        self.daughterOverlapTime = np.min([self.divtime, daughter_tau]) #overlapping time with daughter
        
        self.dedifprob = dedifprob #probability to reverse differentiate
        self.desymprob = desymprob #probability to divide into less differentiated cells
        self.deassprob = deassprob #probability to asymetrically divide into one more differentiated and one less differentiated cell-type        
        
        #custom division rules
        if cusdivprobs is not None and cusdivDaughters is not None and cusdivprobs != 0 and cusdivDaughters != 0:
            assert(len(cusdivDaughters) == len(cusdivprobs))
            self.cusdivprobs = cusdivprobs
            self.cusdivDaughters = cusdivDaughters
        else:
            self.cusdivprobs     = []
            self.cusdivDaughters = []
        
        self.sister     = {}
        self.allsisters = {}
            
        
        self.posunq = {(i + totpts): points[i,:] for i in range(self.ncells)} #position of cell by unique index, allows to keep track of division and differentiation
        
        self.verbose = verbose
                
        if daughterType < 0:
            if self.popidx == 0:
                self.daughterType = 0
            else:
                self.daughterType = self.popidx - 1 #index of the daughter cell population with A->B+B, A->A+B allowed
        else:
            self.daughterType = daughterType

        self.motherType = self.popidx + 1 #index of the mother cell population with A->B+B, A->A+B allowed
        
        self.fixchild = fixchild  #fix child cell position during divisions
        
        
        if thetas is None:
            thetas = np.zeros(self.ncells)
               
        self.angunq = {(i + totpts): thetas[i] for i in range(self.ncells)}
        
        self.desc   = {i: [] for i in range(totpts, totpts + self.ncells)} #descendents
        self.ancs   = pd.Series(data = np.arange(totpts, totpts + self.ncells), index = np.arange(totpts, totpts + self.ncells), 
                                dtype = 'int32') #ancestors: at first time ancestor are themself
        self.parent = {i: i  for i in range(totpts, totpts + self.ncells)} #parents: at first time parents are themself
        
        self.decTime = {} #decision times
        self.dx = dx #distance
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with tuple of parent and daughters
        self.totpts  = totpts + self.ncells #total number of points in history of simulation (used for lineage tracing)
                
        self.div1_dist = None
        self.born = pd.Series(data = -self.first_div_time(self.ncells).astype('int32'), index = np.arange(totpts, totpts + self.ncells), dtype = 'int32') # birth-time        
        if self.divtime > 0:
            initDecTimes = self.first_div_time(self.ncells)
            #initialisation
            for i in range(self.ncells):
                self.decTime.setdefault(next_floor(initDecTimes[i]), []).append(i + totpts)        
        
        
    def clear(self):
        '''
            Remove all cells from the population
        '''
        
        self.posunq = {}
        self.angunq = {}
        
        self.desc   = dict()
        self.ancs   = dict() #ancestors: at first time ancestor are themself
        self.parent = dict() #parents: at first time parents are themself
        self.born   = pd.Series(dtype = 'int32') #birth-time
        
        self.decTime = {} #decision times
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with t  
        self.daughter = {} #stores daughters of cells
        self.sister = {}
        self.alldaughters = {}      
        self.allsisters = {}

        self.ncells = 0
        

    def reset_dectime(self, d, oldtime):
        ''' 
            Reset decision time for a cell
        '''
        #self.decTime[oldtime].remove(d)
        if self.divtime > 0:
            dtimes = self.lifetime()
            self.decTime.setdefault(next_floor(dtimes[0]), []).append(d)

    
    def mutate(self, popidx, params, popname = None):
        '''
            Mutate the population by copying and modifying particular 
                parameters, leaving a population with no cells
                
            To add cells to population use fixedMutate or randomMutate
        '''
        
        new_pop = self.copy()
        new_pop.clear()
        
        new_pop.popidx = popidx
        
        if popname is not None:
            new_pop.popname = popname
        else:
            if isinstance(new_pop.popname, str):
                new_pop.popname += '*'
            else:
                new_pop.popname = str(popidx)
        
        #Update parameters to the mutated population
        for key in params.keys():
            setattr(new_pop, key, params[key])

        return new_pop
        
    
        
    def lifetime(self, ncells = 1, k = 1.0, drate = None):
        '''
            Generate lifetime of ncells from population
            
            k is a multiplying parameter for the shape factor
            
            Note: we use a continuous variable, which we take the 
                floor and add one when setting the division time
        '''
        if drate is None:
            drate = self.divtime
        
        if drate <= 0:
            return np.zeros(ncells)
        elif drate < 1:
            return np.ones(ncells)
        
        if self.cv_div == 1.0:
            return np.random.exponential((drate - 1)*k, ncells)
        else:
            return np.random.gamma(k/self.cv_div**2, (drate - 1)*self.cv_div**2, ncells)
        
    
    def sister_lifetimes(self, corr, drate = None):
        '''
            Generates correlated sister lifetimes for symmetric 
        '''
        
        if corr == 0:
            return self.lifetime(2)
        else:
            return self.lifetime(1, corr, drate) + self.lifetime(2, 1 - corr, drate)
      
      
    def assym_sister_lifetimes(self, corr, tiss):
        '''
            Generates correlated sister lifetimes for asymmetric division
        '''
        if corr == 0:
            dtimes = np.zeros(2)
            dtimes[0] = self.lifetime()
            dtimes[1] = tiss.cellpops[self.daughterType].lifetime()
        else:
            dtimes = self.sister_lifetimes(corr, self.daughterOverlapTime)
            dtimes[0] += self.lifetime(drate = self.divtime - self.daughterOverlapTime)
            dtimes[1] += self.lifetime(drate = tiss.cellpops[self.daughterType].divtime - self.daughterOverlapTime)
        return dtimes
    
    def first_div_time(self, ncells):
        '''
            Generate the first division time for cells, 
                This corresponds to 1 - Cdf(t) after the refractory period 
                which is characterised by the regularised incomplete gamma function  
        '''
        
        if self.divtime <= 1:
            return np.zeros(ncells) 
        elif self.cv_div == 1.0:
            return np.random.exponential((self.divtime - 1), ncells)
        else:
            if self.div1_dist is None:
                self.div1_dist = BornDistribution(1/self.cv_div**2, (self.divtime - 1)*self.cv_div**2, self.refract)
            return self.div1_dist.rvs(ncells)
           
           
    def end_sim(self, **kwargs):
        '''
            Ending simulation, cleans up things for saving:
        '''
        self.div1_dist = None
           
        
    def copy(self):
        return copy.deepcopy(self)    
        
    def getIndices(self): #gets arrays of cell indices for alive cells
        return np.array(list(self.posunq.keys()))
    
    def getPoints(self): #gets array of cell-centre positions
        return np.array(list(self.posunq.values()))
        
    def getThetas(self): #gets array of cell-centre angular orientations
        return np.array(list(self.angunq.values()))
    
    def updateTotpts(self, totpts):
        self.totpts = totpts
        return totpts
    
    def getTotpts(self):
        return self.totpts
    
    # update division probabilities according to some rules
    def updateProb(self):
        return self.divprob
        
    # update points, for class
    def updatePoints(self, points):
        self.posunq = dict(zip(self.posunq.keys(), points)) #find a faster way to do this
        return self.posunq

    # update thetas, for class    
    def updateThetas(self, thetas):
        self.angunq = dict(zip(self.angunq.keys(), thetas)) #find a faster way to do this
        return self.angunq
    
    
    def reset(self, oldtime = 0):
        self.basereset(oldtime)
    
    
    def basereset(self, oldtime = 0):
        #Reset division and born times
        
        self.dupTime.clear()
        self.dieTime.clear()
        self.dupTime.clear()

        if oldtime == 0: #complete reset
            self.born = pd.Series(data = -self.first_div_time(self.ncells).astype('int32'), index = self.getIndices(), dtype = 'int32') # birth-time        
            
            self.decTime.clear()
            if self.divtime > 0:
                initDecTimes = self.first_div_time(self.ncells)
                #initialisation
                for i in range(self.ncells):
                    self.decTime.setdefault(next_floor(initDecTimes[i]), []).append(self.getIndices()[i])                

                self.born = pd.Series(data = 0, index = self.getIndices(), dtype = 'int32')

        else: #Partial reset
            #reset birth times
            self.born -= oldtime

            if  self.divtime > 0 or len(self.decTime) > 0:
                self.decTime = dict(zip(np.array(list(self.decTime.keys())) - oldtime, self.decTime.values()))
                
                deadtimes = np.array(list(self.decTime.keys()))[np.array(list(self.decTime.keys())) <= 0]
                deadcells = []
                for dt in deadtimes:
                    deadcells += self.decTime[dt]
                
                #print(deadcells, self.popidx)
                intersection = [value for value in self.getIndices() if value in deadcells]
                
                if len(intersection) > 0:
                    print('Warning: Decided cells in population', self.popidx)
                    #assert(len(intersection) == 0)
                
        #Reset ancestors
        self.ancs   = {i: i  for i in self.getIndices()} #ancestors: at first time ancestor are themself
    
    
    #make cell-decisions:
        #Implement different strategies
    def cellDecision(self, time, tiss = None): #tissue is not needed
        self.stochasticDecision(time, tiss)
        
        
    def stochasticFateChoice(self, d, tiss, time, randfate = -1):
        if d not in list(self.posunq.keys()): #may have made a decision by some other means
            if self.verbose:
                print('Warning: cell', d, 'does not exist')
            return False

        if randfate <= self.divprob: #replace with: self.divprob[i] # divide
            success = self.divide(d, time, tiss)  
                
        elif randfate <= self.divprob + self.dedifprob:             
            if self.verbose:
                print(d, 'reverse differentiates', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob, 'at', np.round(self.posunq[d],2))
            assert(self.popidx < tiss.npops)

            success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)

            
        elif randfate <= self.divprob + self.dedifprob + self.assprob:
            if self.verbose:
                print(d, 'ass divs', randfate, self.divprob + self.dedifprob +  self.assprob, 'to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2))
            success = self.assymDivide(d, time, tiss)
            
        elif randfate <= self.divprob + self.dedifprob + self.assprob + self.terprob:
            if self.verbose:
                print(d, 'term divides', randfate, self.divprob + self.dedifprob + self.assprob + self.terprob, 'at', np.round(self.posunq[d],2))
            
            success = self.termDivide(d, tiss, time)  
    
        elif randfate <= self.divprob + self.dedifprob + self.assprob + self.terprob + self.difprob:
            success = self.differentiate(d, time, tiss)
            
        elif randfate <= self.divprob + self.dedifprob + self.assprob + self.terprob + self.difprob + self.desymprob:
                if self.verbose:
                    print(d, 'reverse divides', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob, 'at', np.round(self.posunq[d],2))
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
            
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob + self.difprob + self.desymprob + self.deassprob:
                if self.verbose:
                    print(d, 'reverse assdivides', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob, 'at', np.round(self.posunq[d],2))
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)            
            
        elif randfate <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob + np.sum(self.cusdivprobs):
            p0 = self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob
            rf1 = randfate - p0

            for p in range(len(self.cusdivprobs)):
                if rf1 < np.cumsum(self.cusdivprobs[p]):
                    if self.verbose:
                        print(d, 'custom divides', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob + np.sum(self.cusdivprobs), 'at', np.round(self.posunq[d],2), 'to', self.cusdivDaughters[p])
                    success = self.customDivide(d, tiss, time, daughter1type = self.cusdivDaughters[p][0], daughter2type = self.cusdivDaughters[p][1])
                    break

        else: #die
            if self.verbose:
                print(d, 'delaminates', randfate, self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob + np.sum(self.cusdivprobs), 'at', np.round(self.posunq[d],2))
            
            success = self.die(d, time)  

        return success
    
    def divideFateChoice(self, d, tiss, time, randfate = -1):
        return self.stochasticFateChoice(d, tiss, time, randfate)
    

    def stochasticDecision(self, time, tiss = None):
        if time in self.decTime:
            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            if self.verbose:
                print('Stochastic: Pop', self.popidx, 'cells', self.decTime[time], 'are deciding at time', time)
            
            for i in range(ndec):
                d = self.decTime[time][i]
                self.divideFateChoice(d, tiss, time, randfate = rnums[i])
        
                  
    
    def divide(self, d, time, tiss = None): # cell d divides at time t
        success = False
        if d in list(self.posunq.keys()): #check the cell index is valid
            if self.verbose:
                print(d, 'divides into', [self.totpts, self.totpts + 1], 'at', np.round(self.posunq[d],2))
                
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []
            self.desc[self.totpts + 1] = []    

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.ancs[self.totpts + 1] = self.ancs[d]
            
            self.parent[self.totpts]     = d
            self.parent[self.totpts + 1] = d

            self.born[self.totpts]     = time
            self.born[self.totpts + 1] = time

            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                theta = np.random.uniform(2*np.pi)
                self.posunq[self.totpts]     = posd                
                self.angunq[self.totpts]     = angd
                self.angunq[self.totpts + 1] = angd                  
            else:
                self.posunq[self.totpts]     = posd + dv
                self.angunq[self.totpts]     = theta
                self.angunq[self.totpts + 1] = np.pi - theta

            self.posunq[self.totpts + 1] = posd - dv
            

            #set decision times
            if self.divtime > 0:
                dtimes = self.sister_lifetimes(self.corrSym) + time + self.refract 
                self.decTime.setdefault(next_floor(dtimes[0]), []).append(self.totpts)
                self.decTime.setdefault(next_floor(dtimes[1]), []).append(self.totpts + 1)
            
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            #update number of points
            self.totpts += 2
            self.ncells += 1
            
            success = True

        return success


    
    def die(self, d, time = 0): # cell d dies
        success = False
           
        if d in list(self.posunq.keys()):
            if self.verbose:
                print('Removing', d, 'from', self.popidx, 'at', time, 'at', np.round(self.posunq[d],2)) #self.cellidx
        
            #remove particle
            self.posunq.pop(d)
            self.angunq.pop(d)
            self.dieTime.setdefault(time, []).append(d) # self.dupTime
            self.ncells -= 1

            success = True            

        return success

    def deleteRandom(self, time): #pick a cell at random from population to detete
        if self.ncells > 1:
            toDel = np.random.randint(0, self.ncells)
            #print(toDel, self.ncells)
            self.die(list(self.posunq.keys())[toDel], time)
        else:
            print('Cells left:', self.posunq)
            
         
    def deleteOther(self, d, time = 0): # select other cell at random and kill
        if self.ncells > 1:
            toDel = np.random.randint(0, self.ncells - 1)
            if toDel >= d:
                toDel += 1
            self.die(list(self.posunq.keys())[toDel], time)
        else:
            print('Cells left:', self.posunq)
       
    
    
    def differentiate(self, d, time, tiss, daughterType = None):
        success = False    
        if d in list(self.posunq.keys()): #check the cell index is valid
            if daughterType is None:
                daughterType = self.daughterType

            #set descendents
            self.desc[d] = [self.totpts]

            if self.verbose:
                print(d, 'differentiates from', self.popidx, 'to', daughterType, 'at', time, 'at', np.round(self.posunq[d],2)) #self.cellidx           

            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #store time of decision
            self.dupTime.setdefault(time, []).append([d, self.totpts])
            
            self.ncells -= 1
            
            # add cell of other type
            tiss.addCell(daughterType, self.totpts, posd, angd, self.ancs[d], d)

            #update number of points
            self.totpts += 1
                
            success = True            
       
        return success
           
    
    def setDaughter(self, daughterList = None, tiss = None):
        if daughterList is None:
            daughterList = -np.ones(self.ncells, dtype = np.int32)
        assert(len(daughterList) == self.ncells)
        self.daughter     = dict(zip(list(self.getIndices()), daughterList))
        self.alldaughters = dict(zip(list(self.getIndices()), daughterList))

        if tiss is not None:
            #set the sisters
            for i in range(self.ncells):
                d = daughterList[i]

                if d >= 0:
                    tiss.cellpops[self.daughterType].sister[d]     = list(self.getIndices())[i]
                    tiss.cellpops[self.daughterType].allsisters[d] = list(self.getIndices())[i]                    

            #set escape times for TA cells which do not have sisters
            hassister = tiss.cellpops[self.daughterType].sister.keys()
            idx2 = list(tiss.cellpops[self.daughterType].getIndices())

            #Add division times
            tiss.cellpops[self.daughterType].decTime.clear()
                
            if tiss.cellpops[self.daughterType].divtime > 0:
                divtime = tiss.cellpops[self.daughterType].divtime
            else:
                divtime = self.divtime

            for i in idx2:
                if i not in hassister:

                    #set is own sister, we don't have the pair correlation
                    tiss.cellpops[self.daughterType].sister[i]     = i
                    tiss.cellpops[self.daughterType].allsisters[i] = i                    
                    
                    if divtime > 0:
                        dtimes = self.lifetime(1) + self.refract - tiss.cellpops[self.daughterType].born[i]
                        tiss.cellpops[self.daughterType].decTime.setdefault(next_floor(dtimes[0]), []).append(i)                
                    
                
        #assert(tiss.cellpops[self.daughterType].ncells == 
    
        
    def setParent(self, daughterList, parentList):
        self.parent = dict(zip(daughterList, parentList))


            
    def assymDivide(self, d, time, tiss):
    # assymetric division
    # fixchild = True for daughter cell of same type position fixed
        success = False
        
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.parent[self.totpts] = d
            self.sister[self.totpts] = self.totpts + 1
            self.allsisters[self.totpts] = self.totpts + 1
            
            if self.verbose:
                print(d, 'ass divides from', self.popidx, 'to', self.daughterType, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
            
            self.born[self.totpts] = time
            
            posd = self.posunq[d]
            angd = self.angunq[d]
                
            #remove particle
            self.posunq.pop(d)
            self.angunq.pop(d)            
            
            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                self.posunq[self.totpts]     = posd                
                self.angunq[self.totpts]     = angd    
                            
            else:
                self.posunq[self.totpts]     = posd + dv
                self.angunq[self.totpts]     = theta

            divtimes = self.assym_sister_lifetimes(self.corrAss, tiss) 
            #set decision times
            if self.divtime > 0:
                dtimes = divtimes[0] + time + self.refract
                self.decTime.setdefault(next_floor(dtimes), []).append(self.totpts)
                        
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            

            dtime = None
                        
            #add extra cell to the tissue
            tiss.addCell(self.daughterType, self.totpts + 1, posd - dv, np.pi - theta, self.ancs[d], d, self.totpts, dtime = dtime, drate = 0, delay = divtimes[1])    
               
            
            ogtotpts = self.totpts
            
            #update number of points
            #self.ncells += 1
            self.totpts += 2    

            success = True            
            
        return success   

    def customDivide(self, d, tiss, time, daughter1type = None, daughter2type = None): 
    # "terminal" division of cell-type A into cell-type B
        
        if daughter1type is None:
            daughter1type = self.daughterType
            
        if daughter2type is None:
            daughter2type = self.daughterType        
        
        success = False
       
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            if self.verbose:
                print(d, 'term divides from', self.popidx, 'to', daughter1type, ',',  daughter2type, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
                
            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 
                            
            else:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            dtime = None
            delay = 0
            drate = None
                                
            if np.max([daughter1type, daughter2type]) != tiss.npops:
                sisA = self.totpts + 1
                sisB = self.totpts   
            else:
                sisA = -1
                sisB = -1
                
            tiss.addCell(daughter1type, self.totpts, pos1, ang1, self.ancs[d], d, 
                sister = sisA, dtime = dtime, delay = delay, drate = drate)                   
            tiss.addCell(daughter2type, self.totpts + 1, pos2, ang2, self.ancs[d], d, 
                sister = sisB, dtime = dtime, delay = delay, drate = drate) 
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success  
    
    def termDivide(self, d, tiss, time, daughterType = None): 
    # "terminal" division of cell-type A into cell-type B
        
        if daughterType is None:
            daughterType = self.daughterType
        
        success = False
       
        if d in list(self.posunq.keys()): #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            if self.verbose:
                print(d, 'term divides from', self.popidx, 'to', daughterType, 'giving', self.desc[d], 'with ancestor', self.ancs[d], 'at', np.round(self.posunq[d],2))
                
            #remove particle
            posd = self.posunq[d]
            self.posunq.pop(d)

            angd = self.angunq[d]
            self.angunq.pop(d)

            #add new particles positions using random orientation
            theta = np.random.uniform(2*np.pi)
            dv = self.dx*np.array([np.cos(theta), np.sin(theta)])
            
            if self.fixchild:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 
                            
            else:
                pos1 = posd                
                ang1 = angd 
                pos2 = posd - dv
                ang2 = np.pi - theta 

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            dtime = None
            if self.daughterType != tiss.npops:
                sisA = self.totpts + 1
                sisB = self.totpts   
            else:
                sisA = -1
                sisB = -1    


            dtime = tiss.cellpops[self.daughterType].sister_lifetimes(tiss.cellpops[self.daughterType].corrSym)
     
            #popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None
            tiss.addCell(daughterType, self.totpts,     pos1, ang1, self.ancs[d], d, 
                sister = sisA, dtime = None, delay = dtime[0], drate = 0)                   
            tiss.addCell(daughterType, self.totpts + 1, pos2, ang2, self.ancs[d], d, 
                sister = sisB, dtime = None, delay = dtime[1], drate = 0) 
            
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success       
    
    
    def addCell(self, d, pos, theta, time, anc, parent, sister = -1, dtime = None, delay = 0, drate = None):
        #dtime < 0: cell does not divide
        #note: this does not update totpts        
        
        self.posunq[d]     = pos
        self.angunq[d]     = theta

        self.desc[d] = []
        self.ancs[d] = anc
        self.parent[d]  = parent
        self.born[d] = time
       
        self.ncells += 1   
        
        if sister >= 0:
            self.sister[d]  = sister
            self.allsisters[d] = sister
        
        if self.verbose:
            print('Adding', d, 'to pop', self.popidx, 'at', time, 'has ancestor', anc, 'parent', parent, 'at', np.round(self.posunq[d],2)) #self.cellidx     
        
        if drate is None:
            drate = self.divtime #this is a time not a rate

        if dtime is None:
            #set decision times
            if self.divtime > 0:
                dtime = self.lifetime(drate = drate) + time + self.refract + delay
                self.decTime.setdefault(next_floor(dtime[0]), []).append(d)
                #if self.verbose:
                #    print('cell', d, 'divides at rate', self.divtime)
            
        elif dtime > 0:
            self.decTime.setdefault(next_floor(dtime[0]), []).append(d) 
            if self.verbose:
                print('cell', d, 'will divide at', dtime[0], next_floor(dtime[0]), 'see', self.decTime[next_floor(dtime[0])])
        else:
            if self.verbose:
                print('No decision made by cell', d)         
         
        #print(anc, 'makes', d)

    def posToDens(self, perArea = True, centre = False, tiss = None, gridsz = 0, h = None, L = 1, Ly = None, gridszy = None): #maps position vector to density, scaled by cell area
        '''
        if   perArea == True  and self.consumArea == 0.0:
            return np.zeros([gridsz, gridsz])
            
        elif perArea == False and self.consumCell == 0.0:
            return np.zeros([gridsz, gridsz])
        '''
        
        if Ly is None:
            Ly = L
        
        if gridszy is None:
            gridszy = gridsz
        
        if tiss == None or perArea == False:
            areas = np.ones(self.ncells)
        else:
            areas = np.abs(tiss.popAreas(self.popidx))
        
        if h is None:
            h = L/gridsz
        
        if centre:
            rho = np.zeros([gridsz, gridszy])            
            locs = np.floor((self.getPoints() % (np.array([self.L, self.Ly])[None,:]))/h).astype('int')

            for i in range(len(locs)):
                rho[locs[i,0], locs[i,1]] += areas[i]/h**2
            
            
        else: #consumes evenly over cell, by default is proportional to area     
            rho = rasteriseCells(np.arange(tiss.splits[self.popidx], 
                tiss.splits[self.popidx + 1]), tiss.vertices, tiss.regions, 
                tiss.point_region, perArea, h, L, Ly)  

        return rho
        

class sizulation(cellulation):
    ''' 
        cellulation with division rules based on determinisitic dynamics based on 
            size of cell exceeding a critical value, or from random dynmics
            
        To specify different fate decisions rules for size and stochastic decisions use:
            bigDaughter or smallDaughter. This only enables 1 reaction type on each condition.
            If bigDaughter/smallDaughter is an integer, then we differentiate to this population
            Else if a tuple/list of length 2 is provided we use a custom decision of a certain type

            If None is provided, then we use the same rules as divideFateChoice
            
    '''

    def __init__(self, points, divtime = 1000, divprob = 1.0, totpts = 0, thetas = None, 
        dx = 0.01, Alow = 0.0, Ahigh = np.inf,  assprob = 0.0, difprob = 0.0, 
        terprob = 0.0, daughterType = -1, fixchild = True, popidx = 0, areadivtime = -1, refract = 0, 
        profile = 'binary', divhilln = 2, Ascale = 1.0, 
        dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, useRandom = None,
        forceRandFate = False, bigDaughter = None, smallDaughter = None, **extras):
        
        if assprob < 0.0: #use assprob < 0.0 for assymetric division, and probabilistic death
            self.assdiv = True
            assprob = 0.0
        else:
            self.assdiv = False    
    
        cellulation.__init__(self, points, divtime, divprob, totpts, thetas, dx, 
            assprob, difprob, terprob, daughterType, fixchild, popidx, refract, 
            dedifprob = dedifprob, desymprob = desymprob, 
            deassprob = deassprob, **extras)
            
        self.poptype = 'size'
        self.Alow  = Alow
        self.Ahigh = Ahigh
        
        #Set daughter types from 
        self.bigDaughter = bigDaughter
        assert((self.bigDaughter   is None) or (np.ndim(self.bigDaughter)   == 0) or (len(self.bigDaughter)   == 2))
        self.smallDaughter = smallDaughter
        assert((self.smallDaughter is None) or (np.ndim(self.smallDaughter) == 0) or (len(self.smallDaughter) == 2))
        
        if areadivtime > 0:
            self.areadivprob = 1/areadivtime
        else:
            self.areadivprob = 1.0
        
        if useRandom is None:
            self.useRandom = divtime > 0 #whether to use stochastically timed decisions as well as determinitic
        else:
            self.useRandom = useRandom

        self.forceRandFate = forceRandFate #if true then we use a stochastic fate choice

        '''
        For profile of division we have the following options:
            (equivalent in reverse for A < Alow)
            profile = 'binary':  this is just a binary switch, fixed rate = divtime for A < Ahigh, and 1/rate = 1/areadivtime for A > Ahigh
            
            profile = 'relu':  increases linearly above Ahigh, fixed rate = divtime for A < Ahigh, and 1/rate = (A - Ah)/Ah/areadivtime + 1/divtime for A > Ahigh
            
            profile = 'hill':   hill function increasing rate, fixed rate = divtime for A < Ahigh, and 1/rate = hill((A - Ah)/Asc,n) *(1/areadivtime - 1/divtime) + 1/divtime for A > Ahigh
                hill: defaults to Ahigh = 0, n = divhilln = 2, and Asc = Ascale is scaling parameter on hill function
            
            profile = 'dual':      linearly between two rates, fixed rate = divtime for A < Alow, 1/rate = 1/areadivtime for A > Ahigh, linearly interporated in middle
                        
            Note: unless binary is selected then there is no possibility of cells being "too small",
                this is a future possibility to implement, e.g. a profile of delamination

        '''

        self.profile = profile

        if profile == 'relu':
            self.areadivprob /= self.Ahigh

        elif profile == 'hill':
            self.divhilln = divhilln
            self.Ascale      = Ascale #scaling parameter, setting when hill = 0.5, so rate is halfway between each value

            if self.divtime  > 0:
                self.areadivprob = 1/areadivtime - 1/self.divtime
            #else just: 1/self.areadivtime

        elif profile == 'dual':
            self.Aupper = self.Ahigh
            self.Ahigh  = self.Alow
            if self.divtime  > 0:
                self.areadivprob = 1/areadivtime - 1/self.divtime
                
        if self.Alow == 0.0 and self.Ahigh >= 1000:
            self.sizerule = False
        else:
            self.sizerule = True




    def smallprob(self, smallSizes):
        if self.profile == 'binary':
            probs = self.areadivprob
        else:
            probs = 0.0

        return probs
    
    def bigprob(self, bigSizes):
        if self.profile == 'binary':
            probs = self.areadivprob
        elif self.profile == 'relu':
            probs = relu(bigSizes, m = self.areadivprob, x0 = self.Ahigh)
        elif self.profile == 'hill':
            probs = relu(hill((bigSizes - self.Ahigh)/self.Ascale, self.divhilln)*self.areadivprob)
        elif self.profile == 'dual':
            probs = np.ones(len(bigSizes))*self.areadivprob
            probs[bigSizes < self.Aupper] = relu(bigSizes[bigSizes < self.Aupper], m = self.areadivprob/(self.Aupper - self.Ahigh), x0 = self.Ahigh)
        else:
            probs = 0.0
        return probs


    def getSmall(self, area, keys):
        if self.Alow == 0.0 or self.profile == 'dual':
            tooSmall   =  np.array([])
            smallSizes = np.array([])
        elif self.Alow >= 1000:
            tooSmall   = keys
            smallSizes = area
        else:           
            smallidx = np.argwhere(area < self.Alow ).transpose()[0]
            tooSmall   = keys[smallidx]
            smallSizes = area[smallidx]
        return tooSmall, smallSizes
    

    def getBig(self, area, keys):
        if self.Ahigh >= 1000:
            tooBig   =  np.array([])
            bigSizes = np.array([])   
        elif self.Ahigh == 0.0:
            tooBig   = keys
            bigSizes = area          
        else:           
            bigidx = np.argwhere(area > self.Ahigh).transpose()[0]
            tooBig   = keys[bigidx]
            bigSizes = area[bigidx]         
        return tooBig, bigSizes    


    def smallDecision(self, d, time, tiss):
        '''
            Decision that we execute when cells are smaller than a certain area
        '''

        if self.verbose:
            print(d, 'from pop', self.popidx, 'is too small  at ', np.round(self.posunq[d],2), 'time =', time)                    

        
        if self.smallDaughter is None:
            if self.difprob > 0.0:
                self.differentiate(d, time, tiss) 
            elif self.dedifprob > 0.0:
                self.differentiate(d, time, tiss, self.popidx + 1)                        
            else:                                     
                self.die(d, time)       
                
        else:
            if np.ndim(self.smallDaughter) == 0:
                self.differentiate(d, time, tiss, daughterType = self.smallDaughter)
            else:
                self.customDivide(d, tiss, time, self.smallDaughter[0], self.smallDaughter[1])


    def bigDecision(self, d, time, tiss):
        '''
            Decision that we execute when cells are larger than a certain area
        '''
        
        if self.verbose:
            print(d, 'from pop', self.popidx, 'is too big  at ', np.round(self.posunq[d],2), 'time =', time)                    
        
        
        if self.bigDaughter is None:
            self.divideFateChoice(d, tiss, time, randfate = -1)
        else:
            if np.ndim(self.bigDaughter) == 0:
                self.differentiate(d, time, tiss, daughterType = self.bigDaughter)
            else:
                self.customDivide(d, tiss, time, self.bigDaughter[0], self.bigDaughter[1])




    def cellDecision(self, time, tiss):
        if self.sizerule is True:
            #This doesn't work in principle once you change the number of cells
            area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
            keys = tiss.getPopIndices(self.popidx)
            #keys = np.array(list(self.posunq.keys()))
            assert(len(area) == len(keys))


            tooSmall, smallSizes = self.getSmall(area, keys)
            tooBig, bigSizes = self.getBig(area, keys)

            if self.areadivprob >= 1.0:
                for d in tooSmall:
                    if d not in list(self.posunq.keys()):
                        if self.verbose:
                            print(d, 'already divided')
                        continue                
                    
                    if time - self.born[d] > self.refract:   
                        self.smallDecision(d, time, tiss)
                        
                for d in tooBig:
                    if d not in list(self.posunq.keys()):
                        if self.verbose:
                            print(d, 'already divided')
                        continue                
                    
                    if time - self.born[d] > self.refract:                            
                        self.bigDecision(d, time, tiss)
                
            else:
                if len(tooSmall) > 0:
                    rands = np.random.rand(len(tooSmall))
                    for d in tooSmall[rands < self.smallprob(smallSizes)]:
                        if d not in list(self.posunq.keys()):
                            if self.verbose:
                                print(d, 'already divided')
                            continue
                                            
                        if time - self.born[d] > self.refract:
                            self.smallDecision(d, time, tiss)
                
                if len(tooBig) > 0:
                    randb = np.random.rand(len(tooBig))

                    for d in tooBig[randb < self.bigprob(bigSizes)]:
                        if d not in list(self.posunq.keys()):
                            if self.verbose:
                                print(d, 'already divided')
                            continue

                        if time - self.born[d] > self.refract:                
                            self.bigDecision(d, time, tiss)
                        
        if self.useRandom:
            self.stochasticDecision(time, tiss)


    def divideFateChoice(self, d, tiss, time, randfate = -1):
        #    randfate < 0: specifies to apply the rule for large cell-size
        #0 <= randfate <= 1: speciies making the fate decision as for a stochastic fate choice
        # randfate >= 1: forces cell death
        
        if self.forceRandFate is True:
            randfate = np.random.rand(1)
        
        if randfate < 0:
            if self.assdiv: #set for assymetric division
                success = self.assymDivide(d, time, tiss)        
            else:
                if self.terprob > 0.0:
                    if self.verbose:
                        area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                        keys = tiss.getPopIndices(self.popidx)                        
                        areas = dict(zip(keys, area))
                        print(d, 'from pop', self.popidx, 'is big, term divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                    success = self.termDivide(d, tiss, time)
                elif self.dedifprob > 0.0:
                    assert(self.popidx < tiss.npops)
                    success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)
                    
                elif self.difprob > 0.0:
                    success = self.differentiate(d, time, tiss)
                    
                elif self.desymprob > 0.0:
                    if self.verbose:
                        area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                        keys = tiss.getPopIndices(self.popidx)                        
                        areas = dict(zip(keys, area))
                        print(d, 'from pop', self.popidx, 'is big, symmetric de-divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                    success = self.customDivide(d, tiss, time, self.popidx + 1, self.popidx + 1)

                elif self.deassprob > 0.0:                    
                    if self.verbose:
                        area = np.abs(tiss.popAreas(self.popidx))#abs(np.array(tiss.areas[:self.ncells]))
                        keys = tiss.getPopIndices(self.popidx)                        
                        areas = dict(zip(keys, area))
                        print(d, 'from pop', self.popidx, 'is big, asymmetrically de-divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2), 'daughterType', self.daughterType)    
                    success = self.customDivide(d, tiss, time, self.popidx + 1, self.popidx - 1)                    
                
                else:
                    if self.verbose:
                        areas = dict(zip(keys, area))
                        print(d, 'from pop', self.popidx, 'is big, symm divs to', self.totpts, self.totpts + 1, 'at', np.round(self.posunq[d],2), 'time =', time, 'size =', np.round(areas[d],2))    
                    success = self.divide(d, time, tiss)

                    
        else:
            success = self.stochasticFateChoice(d, tiss, time, randfate)
        
        return success
    

class sizulationFR(cellulation):
    #area-based dynamics with fixed stochastic division rate

    def __init__(self, points, rho_c = 1.0, hilln = 2, divtime = 0, totpts = 0, thetas = None, dx = 0.01, 
                 popidx = 0, assprob = 0.0, difprob = 0.0, terprob = 0.0, **extras):
        
        cellulation.__init__(self, points, divtime, totpts = totpts, thetas = thetas, dx = dx, popidx = popidx, assprob = assprob, difprob = difprob, terprob = terprob, **extras)
        
        self.rho_c  = rho_c
        self.hilln  = hilln

        self.poptype = 'size-fixrate'


    def cellDecision(self, time, tiss): 
        #note the probability could be the integral over the trajectory        

        if time in self.decTime:

            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            for i in range(ndec):
                d = self.decTime[time][i]
                if d not in list(self.posunq.keys()):
                    if self.verbose:
                        print(d, 'already divided')
                    continue
                
                area = abs(tiss.areaById(d))
                
                self.divideFateChoice(d, tiss, time, rnums[i], area)
                

    def symmProb(self, d = None, tiss = None, area = None):
        '''
            Probability that cell d executes a duplicative division
                given it choses a symmetric fate decision
        '''
        if area is None:
            area = abs(tiss.areaById(d))
            
        return hill(self.rho_c*area, self.hilln)


    def divideFateChoice(self, d, tiss, time, randfate = -1, area = None):
        #Makes decision based on hill function of cell-area
        if randfate <= -1:
            randfate = np.random.rand(1)
            
        if d not in list(self.posunq.keys()): #may have died from some other means
            if self.verbose:
                print('Warning: cell', d, 'does not exist')
            return False
        
        if area is None:
            area = abs(tiss.areaById(d))        
        
        if randfate < self.assprob:
            success = self.assymDivide(d, time, tiss)
        
        elif randfate < self.assprob + self.deassprob:
            assert(self.popidx < tiss.npops)
            success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)            
            
        
        elif (randfate - (self.assprob + self.deassprob))/(1 - (self.assprob + self.deassprob)) < self.symmProb(d, tiss, area = area):
            if self.desymprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
            else:
               success = self.divide(d, time, tiss)
                            
        else: #differentiate
            if self.difprob > 0.0:
                success = self.differentiate(d, time, tiss)
            elif self.dedifprob > 0.0:
                assert(self.popidx < tiss.npops)
                success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)
            elif self.terprob > 0.0:
                success = self.termDivide(d, tiss, time)
            else:
                success = self.die(d, time)   
                
        return success
    

#non-spatial cellulation
class cellulation0D(cellulation):
    '''
        Cell population class:
            specifies population of cells who have same rules for division vs differentiation
            
            in this version of the code only exponential division with fixed rate and prob is implemented
    '''
    def __init__(self, cellidx = set(), divtime = 1000, divprob = 0.5, totpts = 0, 
                 assprob = 0.0, difprob = 0.0, terprob = 0.0, daughterType = -1, popidx = 0, 
                 refract = 0, motherType = -1, verbose = True,
                 corrSym = 0, corrAss = 0, cv_div = 1.0,
                 cusdivprobs = 0.0, cusdivDaughters = 0.0,
                 dedifprob = 0.0, desymprob = 0.0, deassprob = 0.0, daughter_tau = 0, **extras):
        
        self.poptype = '0D'
        
        self.verbose = verbose
        
        self.popidx = popidx
        
        self.cellidx = set(list(cellidx))
        self.ncells  = len(cellidx) #current number of cells

        self.divtime = divtime  # division time
        self.divprob = divprob  #divprob*np.ones(self.ncells), probability to symmetrically divide
        self.refract = refract #refractory time
        self.cv_div  = cv_div #coefficient of variation of division time
        
        self.assprob = assprob #probability to assymemtrically divide
        self.difprob = difprob #probability to differentiate
        self.terprob = terprob #probability to terminally divide
        
        self.corrSym = corrSym #cell decision correlation for stochastic decisions for sisters born into this population
        self.corrAss = corrAss #cell decision correlation  for stochastic decisions resulting from asymmetric division from this population
        self.daughterOverlapTime = np.min([self.divtime, daughter_tau]) #overlapping time with daughter
        
        self.dedifprob = dedifprob #probability to reverse differentiate
        self.desymprob = desymprob #probability to divide into less differentiated cells
        self.deassprob = deassprob #probability to asymetrically divide into one more differentiated and one less differentiated cell-type           

        #custom division rules
        if cusdivprobs != 0.0 and cusdivDaughters != 0.0:
            assert(len(cusdivDaughters) == len(cusdivprobs))
            self.cusdivprobs = cusdivprobs
            self.cusdivDaughters = cusdivDaughters
        else:
            self.cusdivprobs     = []
            self.cusdivDaughters = []
                
        if daughterType < 0:
            if self.popidx == 0:
                self.daughterType = 0
            else:
                self.daughterType = self.popidx - 1 #index of the daughter cell population with A->B+B, A->A+B allowed
        else:
            self.daughterType = daughterType
        
        if motherType < 0:
            self.motherType = self.popidx + 1
        else: 
            self.motherType = motherType #self.popidx + 1 #index of the mother cell population with A->B+B, A->A+B allowed
                     
        self.desc   = {i: [] for i in range(totpts, totpts + self.ncells)} #descendents
        self.ancs   = {i: i  for i in range(totpts, totpts + self.ncells)} #ancestors: at first time ancestor are themself
        self.parent = {i: i  for i in range(totpts, totpts + self.ncells)} #parents: at first time parents are themself        
        
        
        self.decTime = {} #decision times
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with tuple of parent and daughters
        self.totpts  = totpts + self.ncells #total number of points in history of simulation (used for lineage tracing)
        
        self.div1_dist = None        
        self.born = pd.Series(data = -self.first_div_time(self.ncells).astype('int32'), index = np.arange(totpts, totpts + self.ncells), dtype = 'int32') # birth-time        
        
        if self.divtime > 0:
            initDecTimes = self.first_div_time(self.ncells)
            #initialisation
            for i in range(self.ncells):
                self.decTime.setdefault(next_floor(initDecTimes[i]), []).append(i + totpts)             

        self.futureTimes = SortedSet(list(self.decTime.keys())) #stores division time for future events, useful for simulating longer times

        
    def getIndices(self): #gets arrays of cell indices for alive cells
        return np.array(list(self.cellidx), dtype = np.int32)
        
    def updateTotpts(self, totpts):
        self.totpts = totpts
        return totpts
    
    def getTotpts(self):
        return self.totpts
        
    #make cell-decisions:
        #Implement different strategies
    def cellDecision(self, time, tiss = None): #tissue is not needed
        self.stochasticDecision(time, tiss)
        
        
    def nextEventTime(self):
        if self.ncells > 0 and self.divtime > 0:
            return self.futureTimes[0]
        else:
            return np.inf


    def stochasticDecision(self, time, tiss = None):
        if time in self.decTime:
            ndec = len(self.decTime[time]) #number of decisions
            #based on position calculate division / differentation prob    
            rnums = np.random.random(ndec) # generate random number
            
            if self.verbose:
                print('Stochastic: Pop', self.popidx, 'cells', self.decTime[time], 'are deciding at time', time)
            
            for i in range(ndec):
                d = self.decTime[time][i]
                
                if d not in self.cellidx: #may have died from some other means
                    continue

                if rnums[i] <= self.divprob: #replace with: self.divprob[i] # divide
                    self.divide(d, time)  
                        
                elif rnums[i] <= self.divprob + self.difprob:
                    self.differentiate(d, time, tiss)
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob:
                    self.assymDivide(d, time, tiss)
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob:   
                    self.termDivide(d, tiss, time) 
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob:
                    assert(self.popidx < tiss.npops)
                    success = self.differentiate(d, time, tiss, daughterType = self.popidx + 1)                        
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob:
                        assert(self.popidx < tiss.npops)
                        success = self.customDivide(d, tiss, time, daughter1type = self.popidx + 1, daughter2type = self.popidx + 1)
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob:
                        assert(self.popidx < tiss.npops)
                        success = self.customDivide(d, tiss, time, daughter1type = self.popidx - 1, daughter2type = self.popidx + 1)
                    
                elif rnums[i] <= self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob + np.sum(self.cusdivprobs):
                    p0 = self.divprob + self.difprob + self.assprob + self.terprob + self.dedifprob + self.desymprob + self.deassprob
                    rf1 = rnums[i] - p0

                    for p in range(len(self.cusdivprobs)):
                        if rf1 < np.cumsum(self.cusdivprobs[p]):
                            success = self.customDivide(d, tiss, time, daughter1type = self.cusdivDaughters[p][0], daughter2type = self.cusdivDaughters[p][1])
                            break 

                else: #delaminate
                    
                    success = self.die(d, time) 
                                                               
            assert(self.futureTimes[0] == time)
            self.futureTimes.pop(0)
            
                
                
    def divide(self, d, time, tiss = None): # cell d divides at time t
        success = False

        if d in self.cellidx: #check the cell index is valid
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []
            self.desc[self.totpts + 1] = []    

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.ancs[self.totpts + 1] = self.ancs[d]
            
            self.parent[self.totpts]     = d
            self.parent[self.totpts + 1] = d

            self.born[self.totpts]     = time
            self.born[self.totpts + 1] = time

            #remove particle
            self.cellidx.remove(d)

            #add new particle
            self.cellidx.add(self.totpts)
            self.cellidx.add(self.totpts + 1)  
            
            #set decision times
            if self.divtime > 0:
                dtimes = self.sister_lifetimes(self.corrSym) + time + self.refract

                self.decTime.setdefault(next_floor(dtimes[0]), []).append(self.totpts)
                self.decTime.setdefault(next_floor(dtimes[1]), []).append(self.totpts + 1)
                self.futureTimes.add(next_floor(dtimes[0]))
                self.futureTimes.add(next_floor(dtimes[1]))

            if self.verbose:
                print('Pop', self.popidx, 'cell', d, 'divides into', [self.totpts, self.totpts + 1])
                print([self.totpts, self.totpts + 1], 'will divide at times', [next_floor(dtimes[0]), next_floor(dtimes[1])])
                
            
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            #update number of points
            self.totpts += 2
            self.ncells += 1
            
            success = True

        return success


    
    def die(self, d, time = 0): # cell d dies
        success = False
           
        if d in self.cellidx:
            #remove particle
            self.cellidx.remove(d)            
            self.dieTime.setdefault(time, []).append(d) # self.dupTime
            self.ncells -= 1
            
            if self.verbose:
                print('Removing', d, 'from', self.popidx, 'at', time) #self.cellidx
            
            success = True            

        return success
        

        
    def differentiate(self, d, time, tiss, daughterType = None):
        success = False    
        
        if d in self.cellidx:
            if daughterType is None:
                daughterType = self.daughterType
            
            #set descendents
            self.desc[d] = [self.totpts]

            #remove particle
            self.cellidx.remove(d)

            #store time of decision
            self.dupTime.setdefault(time, []).append([d, self.totpts])
            
            self.ncells -= 1
            
            if self.verbose:
                print(d, 'differentiates from', self.popidx, 'to', daughterType, 'at', time)
                            
            # add cell of other type
            tiss.addCell(daughterType, self.totpts, None, None, self.ancs[d], d)  
            
            #update number of points
            self.totpts += 1
                
            success = True            
       
        return success  
        
        
    def assymDivide(self, d, time, tiss):
        # assymetric division
        # fixchild = True for daughter cell of same type position fixed
        success = False
        
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #set ancestors
            self.ancs[self.totpts] = self.ancs[d]
            self.parent[self.totpts] = d
            
            self.born[self.totpts] = time
        
            #remove particle
            self.cellidx.remove(d)
            
            #add new particle of same type
            self.cellidx.add(self.totpts)
            
            divtimes = self.assym_sister_lifetimes(self.corrAss, tiss) 
            #set decision times
            if self.divtime > 0:
                dtimes = divtimes[0] + time + self.refract
                self.decTime.setdefault(next_floor(dtimes), []).append(self.totpts)
                self.futureTimes.add(next_floor(dtimes))
                        
            if self.verbose:
                print('Pop', self.popidx, 'cell', d, 'ass divides into', [self.totpts, self.totpts + 1])
                print([self.totpts, self.totpts + 1], 'will divide at times', [next_floor(dtimes), next_floor(divtimes[1] + time + tiss.cellpops[self.daughterType].refract)])
                        
                        
            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            #add extra cell to the tissue
            #popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None)
            tiss.addCell(self.daughterType, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, None, delay = divtimes[1], drate = 0)

            #update number of points
            #self.ncells += 1
            self.totpts += 2

            success = True            
            
        return success 


    def customDivide(self, d, tiss, time, daughter1type = None, daughter2type = None): 
        # custom division
        if daughter1type is None:
            daughter1type = self.daughterType
            
        if daughter2type is None:
            daughter2type = self.daughterType   

        success = False
       
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #remove particle
            self.cellidx.remove(d)

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])

            #add extra cells to the tissue
            tiss.addCell(daughter1type, self.totpts,     None, None, self.ancs[d], d, self.totpts, None, 0, None)                   
            tiss.addCell(daughter2type, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, None, 0, None)   
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2   
            
            if self.verbose:   
                print(d, 'custom divides from', self.popidx, 'to', daughter1type, daughter2type, 'giving', self.desc[d], 'with ancestor', self.ancs[d])  
                  
            success = True            

        return success      
    
    def termDivide(self, d, tiss, time): 
    # "terminal" division of cell-type A into cell-type B

        success = False
       
        if d in self.cellidx:
            #set descendents
            self.desc[d] = [self.totpts, self.totpts + 1] 
            self.desc[self.totpts] = []

            #remove particle
            self.cellidx.remove(d)

            #store time of division
            self.dupTime.setdefault(time, []).append([d, self.totpts, self.totpts + 1])
            
            dtime = tiss.cellpops[self.daughterType].sister_lifetimes(tiss.cellpops[self.daughterType].corrSym)
            
            if self.verbose:
                print(d, 'terminally divides from', self.popidx, 'to', self.daughterType, 'giving', self.desc[d], 'with ancestor', self.ancs[d])

            #add extra cells to the tissue
            tiss.addCell(self.daughterType, self.totpts,     None, None, self.ancs[d], d, self.totpts, delay = dtime[0], drate = 0)                   
            tiss.addCell(self.daughterType, self.totpts + 1, None, None, self.ancs[d], d, self.totpts, delay = dtime[1], drate = 0)   
            
            #one less cell of this cell-type
            self.ncells -= 1 
            self.totpts += 2               
                  
            success = True            

        return success       
        
         
        
    def addCell(self, d, pos = None, theta = None, time = 0, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None):
        #note: this does not update totpts       

        self.desc[d] = []
        self.ancs[d] = anc
        self.parent[d]  = parent
        self.born[d] = time
                
        self.cellidx.add(d)
        
        self.ncells += 1   

        if drate is None:
            drate = self.divtime #this is a time not a rate
        
        #set decision times
        if self.divtime > 0:
            dtime = time + self.refract + delay
            if drate > 0:
                dtime += self.lifetime(drate = drate)[0]
            
            self.decTime.setdefault(next_floor(dtime), []).append(d)
            self.futureTimes.add(next_floor(dtime))              

        if self.verbose:
            if dtime is not None:
                print('Adding', d, 'to pop', self.popidx, 'at', time, 'has ancestor', anc, 'parent', parent, 'divtime:', next_floor(dtime)) #self.cellidx
            else:
                print('Adding', d, 'to pop', self.popidx, 'at', time, 'has ancestor', anc, 'parent', parent) #self.cellidx


            

        
    def clear(self):
        '''
            Remove all cells from the population
        '''
        
        self.cellidx = set()
        self.desc   = dict()
        self.ancs   = dict() #ancestors: at first time ancestor are themself
        self.parent = dict() #parents: at first time parents are themself
        self.born   = pd.Series(dtype = 'int32') #birth-time
        
        self.decTime = {} #decision times
        
        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with t        
        self.futureTimes.clear()

        self.ncells = 0
        
    def copy(self):
        return copy.deepcopy(self)
                
            
                
                
    def evolvecopy(self, tsteps, divtime, skip = 10, 
                   deathprofile = None, tstart = 0,
                   useinstantdeath2 = False, born = None,
                   refract = 0):
        ''' 
            Evolve and extract results with a different decay (division) time
            Use instant death kills every 2nd cell instantly then the other comes later: simulating skipping a layer
            Can pass in born times
        '''
        
        #Exclude old cells:
        if born is not None:
            self.born = born
            
        if type(self.born) == dict:
            #included to allow backward compatibility
            self.born = pd.Series({k:v for (k,v) in self.born.items() if v >= tstart})
        else:
            self.born = self.born[self.born >= tstart]
        #self.born = {k:v for (k,v) in self.born.items() if v >= tstart}

        self.cellidx = set() #set(list(self.born.keys()))
        self.ncells = 0 

        self.divtime = divtime
        ncells = len(self.born)

        self.decTime = {}
        if divtime > 0:
            dectime = self.born.values + self.lifetime(ncells) + refract

            for i in range(ncells):
                self.decTime.setdefault(next_floor(dectime[i]), []).append(list(self.born.keys())[i])

        self.dieTime = {} #stores times which cells die, and index of cell
        self.dupTime = {} #stores times which cells divide with t        

        cellids = []
        ncells = np.zeros(tsteps + 1 - tstart, dtype = np.int64)
        #ncells[0] = self.ncells

        birthtime = {}
        for b in self.born.keys():
            birthtime.setdefault(self.born[b], []).append(b)

        #print(birthtime)

        if not useinstantdeath2:
            for time in range(tstart, tsteps + 1):#eventimes:    
                if time in birthtime.keys():
                    #add cells
                    for d in birthtime[time]:
                        #print('Add', d, 'at time', time)                
                        self.cellidx.add(d)
                        self.ncells += 1 
                
                if time in self.decTime:
                    for d in self.decTime[time]:
                        #print('Kill', d, 'at time', time)    
                        success = self.die(d, time)
                        
                ncells[time] = int(self.ncells)
                assert(len(self.cellidx) == self.ncells)
                #print(time, self.cellidx, len(self.cellidx), self.ncells)
                if time % skip == 0: #saves data every 10 time-points
                    cellids.append(list(self.cellidx).copy()) 
        
        elif useinstantdeath2:
            for time in range(tstart, tsteps + 1):#eventimes:    
                if time in birthtime.keys():
                    #add cells
                    #print(birthtime[time])
                    assert(len(birthtime[time]) % 2 == 0)
                    for i in range(len(birthtime[time])):
                        #print('Add', d, 'at time', time)   
                        d = birthtime[time][i] 
                        self.cellidx.add(d)
                        self.ncells += 1                         
                        if i % 2 == 0:
                            success = self.die(d, time)

                if time in self.decTime:
                    for d in self.decTime[time]:
                        #print('Kill', d, 'at time', time)    
                        success = self.die(d, time)                        

                ncells[time] = int(self.ncells)
                assert(len(self.cellidx) == self.ncells)
                #print(time, self.cellidx, len(self.cellidx), self.ncells)
                if time % skip == 0: #saves data every 10 time-points
                    cellids.append(list(self.cellidx).copy())              
        
        if len(cellids) != int((tsteps - tstart)/skip) + 1:
            print('Length is off', len(cellids), int((tsteps - tstart)/skip) + 1)
        
        assert(len(cellids) == int((tsteps - tstart)/skip) + 1)
        
        return {'ncells': ncells, 'cellid': cellids}  
    
            
        
    def reset(self, oldtime = 0):
        self.basereset(oldtime)
        
        decTimeKeys = np.array(list(self.decTime.keys()))        
        self.futureTimes = SortedSet(decTimeKeys[decTimeKeys > 0])
        


class BornDistribution():
    ''' 
        Specify the born distribution and generate random numbers from it
            when lifetimes follow a Gamma distribution
            
        Faster than scipy for generating large numbers of random numbers
        
        See: https://stackoverflow.com/questions/46055690/python-how-to-define-customized-distributions
        
        Lifetime distribution form: exp(-x/scale)*x^{-x/shape}
        This specifies the first event time
        
        Arguments:
              shape: k or alpha
              scale: theta or beta
            refract: refractory period
    '''
    
    def __init__(self, shape, scale, refract):
        
        self.shape = shape
        self.scale = scale
        self.refract = refract
        self.discrete = False
        self._ninvpol = None      
        
        assert(shape >= 0 and scale > 0 and refract >= 0)        
        
    def support(self):
        # distribution restricted to 0, np.inf
        return (0, np.inf)

    def pdf(self, x):
        if self.discrete is True:
            return discrete_born_pdf(x, self.shape, self.scale, self.refract)
        else:
            return born_pdf(x, self.shape, self.scale, self.refract)

    def cdf(self, x):
        return born_cdf(x, self.shape, self.scale, self.refract, self.discrete)

    def rvs(self, size):
        ''' 
            Currently only implemented for continuous lifetime distributions 
        '''
        assert(self.discrete == False)
        return self.ninvpol.rvs(size)*self.scale
    
    @property
    def ninvpol(self): #ninvpol: that is normalisaed
        if self._ninvpol is None:
            scale = self.scale
            self.scale = 1
            self._ninvpol = NumericalInversePolynomial(self)
            self.scale = scale
        return self._ninvpol    


def born_cdf(t, shape, scale, refract, discrete = False):
    # Define the conditions
    conditions = [t < 0, (t >= 0) & (t < refract), t >= refract]
    
    # Define the corresponding functions
    functions = [
        lambda t: 0,
        lambda t: t/(shape*scale + refract),
        lambda t: (refract+int_gamma_reg(t-refract, shape, scale, discrete))/(shape*scale + refract)
    ]
    
    return np.piecewise(t, conditions, functions)


def int_gamma_reg(y, shape, scale, discrete=False):
    if discrete:
        if np.ndim(y) == 0:
            y = int(y)
        else:
            y = y.astype(int)
            
    # Compute Gamma(shape) for the denominator
    Gamma_alpha = scipy.special.gamma(shape)
    
    # Compute the incomplete gamma functions
    Gamma_alpha_y_over_beta = scipy.special.gamma(shape) * scipy.special.gammaincc(shape, y / scale)
    Gamma_alpha_plus_1_y_over_beta = scipy.special.gamma(shape + 1) * scipy.special.gammaincc(shape + 1, y / scale)
    
    # Calculate the expression
    result = shape * scale + (y * Gamma_alpha_y_over_beta - scale * Gamma_alpha_plus_1_y_over_beta) / Gamma_alpha
    return result


def born_pdf(x, shape, scale, refract):
    # Define the conditions
    conditions = [x < 0, (x >= 0) & (x < refract), x >= refract]
    
    # Define the corresponding functions
    functions = [
        lambda x: 0,
        lambda x: 1/(shape*scale + refract),
        lambda x: (1 - scipy.special.gammainc(shape, (x-refract)/scale))/(shape*scale + refract)
    ]
    
    return np.piecewise(x, conditions, functions)


def discrete_born_pdf(x, shape, scale, refract):
    return born_cdf(x+1, shape, scale, refract, True) - born_cdf(x, shape, scale, refract, True)

