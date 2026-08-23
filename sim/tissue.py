import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
from utils import *
from cellulation import *
import copy
import numbers
import pandas as pd
import pickle

#tissue class
class tissue(Voronoi):
    ''' tissue class defines tissue by Voronoi tessalation of points and evolves 
        according to rules of a self-propelled Voronoi model
        
        Calls cellulation class to implement division and differentiation
    '''

    def __init__(self, points = None, thetas = None, L = 1, PBC = True, pop_idx = None, divtime = 0, 
        divprob = 0.0, assprob = 0.0, difprob = 0.0, terprob = 0.0, sbasshed = 0, sbasrefract = 0, sbas_cv = 1.0,
        rule = 'stochastic', poppts = None, seed = None, Ly = None, sim_params = None, 
        reorder_pts = True, **params):
        
        '''

            Arguments for all tissues:
                poppts (array or number): will generate tissue with points at random coordinates
                points (Nx2 array): (x,y) coordinates for N points 
                thetas (Nx1 array): angular orientation of initial velocity, 
                    either None (generated from uniform distribution) or a Nx1 array
                L: linear dimension of box
                PBC: whether to use periodic boundary conditions. Non-PBCs may not work in this version of code
                
            For tissues with division and death:
                For a certain number of cell populations, Each cell is assigned a population index "pop_idx"
                If you want only one cell population, leave this as 0. 
                If you want m populations (for m > 1), set pop_idx as Nx1 array where elements are 0,..., m-1
                    corresponding to index of each population
                For one population:
                    divtime (positive double): is the average division time
                    divprob (double): is the probability to divide (instead of delaminate) 0.0 <= divprob <= 1.0
                For one or more populations:
                    Enter divtime, divprob as lists where values correspond to each population
                    E.g. divtime = [5, 10], divprob = [1.0, 0.0] specifies one renewing and one dying population
                    
                    
            sim_params (dict): dictionary of parameters for simulation, e.g. mu, kA, kP, A0, P0, Dr, v0, dt
                This will fixe the parameters for the simulation, overriding any values passed in
        '''
        
        self.dim = 2 #we have a 2D tissue

        if seed is not None:
            seed = np.random.seed(seed)

        #Set-up initial cells and box
        self.L = L
        if Ly is None:
            Ly = L
        self.Ly = Ly

        if points is not None:
            pop_idx = self.setPoints(points, pop_idx, reorder_pts)
            #self.cellpoints = points #correspond to point locations
        elif poppts is not None:
            pop_idx = self.randomConfig(poppts)
        else:
            assert(0), 'No points or poppts specified'
            
        self.oldpoints  = self.cellpoints.copy()

        
        self.npts   = len(self.cellpoints)
        self.totpts = 0 #total number of cells in history of simulation
        #self.splits = np.array([0, self.npts]) # default splits between points
        self.updateSplits()

        self.E = None
        self.time = 0
                
        if thetas is None:
            self.thetas = np.random.uniform(0, 2*np.pi, self.npts)
        else:
            self.thetas = thetas
        
        #Suprabasal shedding time-scale < 0: no suprabasal layer
        if sbasshed < 0:
            self.suprabasal = False
            self.nsbcells = 0
        else:
            self.suprabasal  = True
            self.sbasshed    = sbasshed
            self.sbasrefract = sbasrefract
            self.sbas_cv     = sbas_cv
                
        
        # Set-up populations
        self.makeParams(pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params)
        self.setupPopulations(pop_idx, divtime, rule, **params)

        #Add present only PBCs are coded
        if PBC:
            self.addPBC()
        else:
            self.fullpts = self.cellpoints
            self.pbcidx = np.arange(self.npts)
        
        #apply PBCs
        self.oldpoints = np.zeros((self.npts,2))
        self.oldnpts = self.npts
        self.updateVoronoi() #take Voronoi tessalation and calculate relevant attributes

        #Set fixed mechanical parameters, overrides values passed in
        self.sim_params = {}
        if sim_params is None:
            self.default_sim_params = {}
        else:
            self.default_sim_params = sim_params


    def setPoints(self, points, pop_idx, reorder_pts = True):
        '''
            Set points and order by population
        '''

        if pop_idx is None:
            pop_idx = np.zeros(len(points), dtype = np.int32)

        if reorder_pts:
            idx = np.argsort(pop_idx, stable = True)
            pop_idx = pop_idx[idx]
            self.cellpoints = points[idx, :]
        else:
            self.cellpoints = points

        self.poppts = np.array([np.sum(pop_idx == i) for i in range(np.max(pop_idx)+1)])

        return pop_idx



    def randomConfig(self, poppts):
        '''
            If input number of cells by population, initialises with randomly places cells:
        '''
             
        if np.ndim(poppts) == 0: #if input as a number
            poppts = [poppts]
        
        npts = np.sum(poppts)

        self.cellpoints = randomConfig(npts, self.L, Ly = self.Ly) #initialisePositions
        self.poppts = poppts
        splits = self.updateSplits()

        popidx = np.zeros(npts, dtype = np.int32)
        for i in range(len(poppts)):
            popidx[splits[i]:splits[i+1]] = i
            
        return popidx


  
    
    def updateVoronoi(self, tolerance = 0.0): #update attributes for changes to points
        #print((self.oldnpts != self.npts), (np.sum(np.abs(self.cellpoints - self.oldpoints)) > tolerance*self.npts), np.sum(np.abs(self.cellpoints - self.oldpoints)), tolerance*self.npts)
                
        #if (self.oldnpts != self.npts) or (np.sum(np.abs(self.cellpoints - self.oldpoints)) > tolerance*self.npts):

        #self.oldpoints = self.cellpoints.copy()
        #self.oldnpts = self.npts
                 
        if self.npts <= 5:
            print('Too few points to calculate Voronoi tessalation')
            return

        self.getVoronoi()
        self.ridge_vertices = np.array(self.ridge_vertices)        
        self.npbcpts   = len(self.fullpts) #number of pbc points 
        self.nv = len(self.vertices)
        self.tri = self.triangles() #updates triangles and interior sites 
        #always update areas, perimeters
        self.edgeLengths = self.ridgeLengths() #check how to correctly use functions to update attributes themself
        
        self.areas = self.cellAreas()
        self.perim = self.cellPerimeters()
        self.jlengths = self.junctionLengths(False, True)
        
        self.adjmat_time = -1
        self._adjmat = None
                    
        #else:
        #    print('Using old points at', self.time)
        

    def getVoronoi(self):
        Voronoi.__init__(self, self.fullpts)
        
        
    def copy(self):
        return copy.deepcopy(self)
        
    
    def reset(self, full = False, verbose = True):
        '''reset's populations retaining final point locations, thetas, 
            cells of each population type and

            Note: the dead cell will be left in decision time, unless a full reset is used.
        '''
        if full == True:
            oldtime = 0
        else:
            oldtime = self.time

        if verbose:
            print('Resetting time and division times')        
        
        for i in range(self.npops):
            self.cellpops[i].reset(oldtime)
            
        self.cellid = self.cellid[:self.npts]
        
        if self.suprabasal:
            self.sbas.reset(oldtime)
            #self.initSuprabasal()
            self.sbas.clear()
            self.nsbcells = 0
                   
        self.time = 0      
        
        self.updateVoronoi() #take Voronoi tessalation and calculate relevant attributes
  
        
    def addPBC(self, rinc = None): #adds periodic boundary conditions
        #rinc is inclusion radius for PBCs
        if rinc == None:
            rinc = self.L

        if rinc < 0:
            rinc = 0.55*np.max(self.perim[:self.npts]) #formerly 1.5 #
            
        vx = self.L *np.array((1,0), dtype = np.float64)
        vy = self.Ly*np.array((0,1), dtype = np.float64)
        boxpoints = self.cellpoints % (np.array([self.L, self.Ly])[None,:])
        left  = boxpoints[:,0] < rinc
        right = boxpoints[:,0] > self.L  - rinc
        up    = boxpoints[:,1] > self.Ly - rinc
        down  = boxpoints[:,1] < rinc
        self.fullpts = np.vstack([boxpoints, 
                            boxpoints[left ].copy() + vx, 
                            boxpoints[right].copy() - vx, 
                            boxpoints[up   ].copy() - vy, 
                            boxpoints[down ].copy() + vy, 
                            boxpoints[left  & down].copy() + vx + vy, 
                            boxpoints[right & up  ].copy() - vx - vy, 
                            boxpoints[left  & up  ].copy() + vx - vy, 
                            boxpoints[right & down].copy() - vx + vy])
                            
        self.PBC  = True
        self.npbcpts = len(self.fullpts)
        
        #store indices
        self.pbcidx = np.concatenate([np.array(np.arange(self.npts)), 
                                    np.nonzero(left)[0], np.nonzero(right)[0],
                                    np.nonzero(up)[0], np.nonzero(down)[0], 
                                    np.nonzero(left  & down)[0], np.nonzero(right & up)[0],
                                    np.nonzero(left  & up)[0], np.nonzero(right & down)[0]])
        
        
        return self.fullpts
    
    """
    def ridgeLengths(self): #calculates lengths of ridges around points
        '''
            Stores the lengths of edges by vertex indices
        '''

        return ridgeLengths(self.vertices, self.ridge_vertices, self.ridge_points, self.npbcpts)
    """  
     
     
    def ridgeLengths(self): #calculates lengths of ridges around points
        '''
            Stores the lengths of edges by vertex indices
        '''
        #use ridge_vertices as np.array
        dist = np.linalg.norm(self.vertices[self.ridge_vertices[:,0],:] - self.vertices[self.ridge_vertices[:,1],:], axis = 1)
        rl = np.zeros((self.nv + 1, self.nv + 1), dtype = np.float64)
        rl[self.ridge_vertices[:,0], self.ridge_vertices[:,1]] = dist
        rl[self.ridge_vertices[:,1], self.ridge_vertices[:,0]] = dist
        return rl       
    


    def junctionLengths(self, incpbc = False, useEdgeLengths = False): #calculates lengths of ridges around points
        '''
            Stores the lengths of edges by cell indices
            
            jl[:npts, :npts] 
            
            if useEdges is True, we assume self.edgeLengths = self.ridgeLenghts() is up to date
        '''
        
        if incpbc:
            jl = np.zeros((self.npbcpts, self.npbcpts), dtype = np.float64)        
            ridge_points = self.ridge_points
            ridge_vertices = self.ridge_vertices
        else:
            jl = np.zeros((self.npts, self.npts), dtype = np.float64)
            incpts = np.min(self.ridge_points,1) < self.npts
            ridge_vertices = self.ridge_vertices[incpts,:]
            #boundary_cell = np.max(self.ridge_points,1) >= self.npts
            ridge_points   = self.pbcidx[self.ridge_points[incpts,:]]
            

        if useEdgeLengths: #this is a vxv matrix
            jl[ridge_points[:,0], ridge_points[:,1]] = self.edgeLengths[ridge_vertices[:,0],ridge_vertices[:,1]]
            jl[ridge_points[:,1], ridge_points[:,0]] = self.edgeLengths[ridge_vertices[:,0],ridge_vertices[:,1]]
        else:
            dist = np.linalg.norm(self.vertices[ridge_vertices[:,0],:] - self.vertices[ridge_vertices[:,1],:], axis = 1)
            jl[ridge_points[:,0], ridge_points[:,1]] = dist
            jl[ridge_points[:,1], ridge_points[:,0]] = dist

        #introduces mistakes on the boundary due to double counting
        #jl += jl.transpose()

        return jl
    
    def jlengths_fullindices(self, a, b):
        '''
            Returns the junction length specified by full inidices a, b
        '''
        return self.jlengths[self.inverseid[a], self.inverseid[b]]

        
    def cellAreas(self): #signed area for all cells
        return [PolyArea(self.vertices[self.regions[v]], np.array(self.regions[v])) for v in self.point_region[:self.npts]]
    
    '''
    def cellPerimeters(self): #perimeters for all cells
        return PolyPerim(self.edgeLengths[:self.npts, :])#, self.ridge_points)
    ''' 
    
    def cellPerimeters(self): #perimeters for all cells
        return [PolyPerim(self.edgeLengths, np.array(self.regions[v]))  for v in self.point_region[:self.npts]]
        
    def areaById(self, i):
        return self.areas[self.inverseid[i]]
        
    def popAreas(self, i): #return the area of all cells in a specified population
        self.updateSplits()
        return np.array(self.areas[self.splits[i]:self.splits[i+1]])
    
    def popPerims(self, i): #return the area of all cells in a specified population
        self.updateSplits()
        return self.perim[self.splits[i]:self.splits[i+1]]    
    
    def updateSplits(self):
        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)  
        self.splits = np.array(self.splits)
        return self.splits
    

    def pop_by_indices(self, idxs, fullid = True):
        '''
            Given list of indices, returns the populations of each index

            i.e. vectorised version of pop_by_idx

            fullid = True: takes in and returns the unique cell index
            fullid = False: takes in and returns the cell position index
        '''

        if fullid:
            idxs1 = np.array(self.inverseid[idxs])
        else:
            idxs1 = idxs

        assert np.min(idxs1) < self.splits[-1] , 'Index out of range ' + str(idxs)

        if np.ndim(idxs) == 0:
            return np.argmax(np.array(self.splits) > idxs1) - 1
        else:
            return np.argmax(np.array(self.splits)[:,None] > idxs1[None,:],0) - 1

    
    def triangles(self): #triangles of points
        #stores point indices for each vertex
        #last element corresponds to points which lie in boundary regions  
        #updates interior sites
        tri = [set() for i in range(self.nv + 1)] 
        for i in range(len(self.ridge_vertices)):
            tri[self.ridge_vertices[i,0]].add(self.ridge_points[i,0])
            tri[self.ridge_vertices[i,0]].add(self.ridge_points[i,1])
            tri[self.ridge_vertices[i,1]].add(self.ridge_points[i,0])
            tri[self.ridge_vertices[i,1]].add(self.ridge_points[i,1])
        
        if np.max([len(tt) for tt in tri[:-1]]) == 4:
            print('WARNING: 4 fold vertices detected, repairing cell positions with noise at time ' + str(self.time))
            self.cellpoints += (np.random.random((self.npts,2)) - 0.5)*1e-7
            self.addPBC()
            self.updateVoronoi()
            return self.tri
            
        self.interior = self.interiorSites(tri)

        return np.array([list(t) for t in tri[:-1]])
    
    def interiorSites(self, triList): #returns interior sites as true and exterior as false
        interior = np.ones(self.npbcpts, dtype = bool)
        interior[np.array(list(triList[-1]))] = False
        return interior
    
    def get_dhdr(self): #all_dhdr(points, tri, npts, nv, vertices)
        #return all_dhdr(self.fullpts, self.tri, self.npbcpts, self.nv, self.vertices)
        return all_dhdr(self.fullpts, self.tri, self.npts, self.nv, self.vertices)

    
    #For optimised code:
    def get_dEdh(self, KA, KP, A0, P0): 
        return all_dEdh(self.vertices, self.regions, self.point_region, self.points, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior, self.pbcidx, self.npbcpts, self.ridge_points, np.array(self.splits))
        #return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior)
    
    '''
    #For old code
    def get_dEdh(self, KA, KP, A0, P0): #all_dEdh(vertices, regions, point_region, Ac, Pc, KA, KP, A0, P0, npts, interior)
        return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, 
                        A0, P0, self.npbcpts, self.interior, self.pbcidx)
        #return all_dEdh(self.vertices, self.regions, self.point_region, self.areas, self.perim, KA, KP, A0, P0, self.npts, self.interior)        
    '''
    
    def getForces(self, dEdh, dhdr): #computeForces(ridge_points, ridge_vertices, point_region, dEdh, dhdr, npts, interior)
        #return computeForces(self.ridge_points, self.ridge_vertices, self.point_region, dEdh, dhdr, self.npts, self.interior)
        return computeForces(self.ridge_points, self.ridge_vertices, self.point_region, dEdh, dhdr, self.npts, self.interior, self.pbcidx)    
            
    def backInBox(self): #move points back in box
        return self.cellpoints % self.L
    

                    
    def rotDiff(self, Dr, dt, v0): #add rotational diffusion (perstent RW of centres)
        #Dr is rotational diffusion coefficient, dt is time-step, v0 is cell velocity
        #further major optimisation from pre-generating random numbers
        
        if Dr > 0.0:
            Drdt = np.sqrt(2*Dr*dt)
            self.thetas += Drdt*np.random.normal(0, 1, self.npts) #Add rotationaldiffusion

        if v0 > 0.0:
            self.cellpoints[:,0] += v0*dt*np.cos(self.thetas)
            self.cellpoints[:,1] += v0*dt*np.sin(self.thetas)
        
        return self.cellpoints
    
    def rotDiffMulti(self, Dr, dt, v0): #add rotational diffusion (perstent RW of centres)
        #Dr is rotational diffusion coefficient, dt is time-step, v0 is cell velocity
        #further major optimisation from pre-generating random numbers

        for i in range(self.npops):
            if self.poppts[i] == 0:
                continue
            
            if Dr[i] > 0.0:
                Drdt = np.sqrt(2*Dr[i]*dt)
                self.thetas[self.splits[i]:self.splits[i+1]] += Drdt*np.random.normal(0, 1, self.poppts[i]) #Add rotationaldiffusion
                     
            if v0[i] > 0:
                self.cellpoints[self.splits[i]:self.splits[i+1],0] += v0[i]*dt*np.cos(self.thetas[self.splits[i]:self.splits[i+1]])
                self.cellpoints[self.splits[i]:self.splits[i+1],1] += v0[i]*dt*np.sin(self.thetas[self.splits[i]:self.splits[i+1]])
        
        
        return self.cellpoints    
        
    def addForces(self, mu, dt, KA, KP, A0, P0):
        #add forces onto each cell
        dhdr   = self.get_dhdr()
        dEdh   = self.get_dEdh(KA, KP, A0, P0)
        forces = self.getForces(dEdh, dhdr)  #computeForce vector
        self.oldpoints = self.cellpoints.copy()
        self.cellpoints += mu*forces[:self.npts, :]*dt
        return self.cellpoints
        
        
    def addForcesMulti(self, mu, dt, KA, KP, A0, P0):
        # Add forces for multiple cell populations
        
        dhdr   = self.get_dhdr()
        dEdh   = self.get_dEdh(KA, KP, A0, P0)
        forces = self.getForces(dEdh, dhdr)  #computeForce vector
        
        self.oldpoints = self.cellpoints.copy()  #save old points      

        for i in range(self.npops):
            if len(self.cellpoints[self.splits[i]:self.splits[i+1],0]) != len(forces[self.splits[i]:self.splits[i+1],0]):
                print(i, self.splits, np.shape(self.cellpoints), np.shape(forces))
            assert(len(self.cellpoints[self.splits[i]:self.splits[i+1],0]) == len(forces[self.splits[i]:self.splits[i+1],0]))
            self.cellpoints[self.splits[i]:self.splits[i+1],:] += mu[i]*forces[self.splits[i]:self.splits[i+1], :]*dt
        
        return self.cellpoints        
        
        
        
    def energy(self, kA, A0, kP, P0): #energy per cell
        if self.npops == 1:
            return np.sum((kA*(abs(np.array(self.areas[:self.npts])) - A0)**2 + kP*(np.array(self.perim[:self.npts]) - P0)**2))/self.npts
        else: #multiple populations
            E = 0
            for p in range(self.npops):
                E += np.sum((kA[p]*(abs(np.array(self.areas[self.splits[p]:self.splits[p+1]])) - A0[p])**2 + kP[p]*(np.array(self.perim[self.splits[p]:self.splits[p+1]]) - P0[p])**2))/self.npts
            return E
    
    def PBCevolve(self, Dr, v0, mu, kA, kP, A0, P0, dt = 1e-2, plot = False): 
        #evolve SPV model one time-step with periodic boundary conditions
        if self.npts == 0:
            return
        
        if (np.ndim(mu) == 0 and mu == 0.0) or (np.ndim(mu) > 1 and np.all(mu == 0.0)):
            return
        
        #0.5 is guaranteed not to break
        rinc = 0.55*np.max(self.perim[:self.npts]) #formerly 1.5 #

        #self.addPBC()        
        self.addPBC(rinc)
        self.updateVoronoi()                             
        
        if self.npops > 1:
            self.addForcesMulti(mu, dt, kA, kP, A0, P0)
            self.rotDiffMulti(Dr, dt, v0)
        else:
            self.addForces(mu, dt, kA, kP, A0, P0)
            self.rotDiff(Dr, dt, v0)
                    
        self.E = self.energy(kA, A0, kP, P0)
        
        if plot: #plot for time-steps
            fig = voronoi_plot_2d(self, show_vertices=False)
            plt.xlim([0, self.L])
            plt.ylim([0, self.Ly])    
            plt.show()

    
    def makeParams(self, pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params):
        keys = ['pop_idx', 'divtime', 'divprob', 'assprob', 'difprob', 'terprob', 'rule'] 
        values = [pop_idx, divtime, divprob, assprob, difprob, terprob, rule]             
        self.params = {**params, **dict(zip(keys, values))}
        return self.params       
    
    #From parameters dictionary, get the parameters for the i-th population
    #This corresponds to taking the i-th element of parameters that are lists
    #Future add for suprabasal layer
    def popParams(self, i):
        popParams = {}
        if i < self.npops: 
            for key in self.params:
                if type(self.params[key]) is list or type(self.params[key]) is np.ndarray:
                    if len(self.params[key]) > self.npops: #don't include variables which are specified by point, e.g. pop_idx
                        continue
                    assert len(self.params[key]) == self.npops , key
                    popParams[key] = self.params[key][i]
                else:
                    popParams[key] = self.params[key]

        #Pass the daughter division time in, useful for enforcing sister correlations
        if i > 0:
            popParams['daughter_tau'] = self.params['divtime'][i-1]
        else:
            if self.suprabasal:
                popParams['daughter_tau'] = self.sbasshed
            else:
                popParams['daughter_tau'] = 0
        
        

        return popParams


    
    def setupPopulations(self, pop_idx, divtime, rule, **params):   
        if self.time > 0:
            self.time = 0
            print('Time reset to 0')
        
        # if we have division and differentiation
        self.npops = 1
        if type(divtime) is list:
            self.npops = len(divtime)
            if hasattr(self, 'poppts'):
                assert(self.npops == len(self.poppts))
            
        #total number of points in each cell population:
        self.poppts = np.zeros(self.npops, dtype = np.int32)  
        
        self.cellpops = [] #list of cell populations, using cellulation objects

        self.makePopulations(pop_idx, rule, **params)

        #Initialise suprabasal population             
        if self.suprabasal:
            self.initSuprabasal()   
        



    def makePopulations(self, pop_idx, rule = 'stochastic', daughterTypes = None, **params):
    
        '''
            Re-orders points, thetas, so one population lies ahead of other populations
            and creates "cellulation" object for each population
            
            
            rule: choose one from {'stochastic', 'size', 'density'}
                
        '''
        if type(rule) is not list:
            rule = [rule for i in range(self.npops)]


        #Parameters that can be same across populations (e.g. corrtimeSym) is deprecated
        # Now must specify for each population, or defaults to all zero

        #Set-up cell points and thetas
        if self.npops > 1:
            parts = partitionTypes(pop_idx, nclasses = self.npops)
        else:
            parts = [np.arange(self.npts)]
        
        if self.dim > 0:
            newpoints = np.reshape(np.array([]), (0,2))
            newthetas = np.array([])
        
        for i in range(self.npops):
            if self.npops == 1:
                if self.dim > 0:
                    poppoints = self.cellpoints
                    popthetas = self.thetas
                    newpoints = poppoints
                    newthetas = popthetas
                else:
                    poppoints = None
                    popthetas = None             
                self.poppts[0] = self.npts

            else:
                poppoints = np.zeros((0,self.dim))
                popthetas = np.zeros((0,self.dim))                 
                if self.dim > 0:
                    if len(parts[i]) > 0:
                        poppoints = self.cellpoints[np.array(parts[i]), :]
                        popthetas = self.thetas[np.array(parts[i])]
                        newpoints = np.vstack([newpoints, poppoints])
                        newthetas = np.hstack([newthetas, popthetas])
                                        
                self.poppts[i] = len(parts[i])
            

            #Set daughter populations
            if type(daughterTypes) is list:
                assert(len(daughterTypes) == self.npops)
                daughterType = daughterTypes[i]
            else:
                if i == 0 and self.suprabasal is True:
                    daughterType = self.npops
                else:
                    daughterType = -1
            
            self.addPopulation(i, rule[i], daughterType, self.poppts[i], poppoints, popthetas)

        if self.dim > 0:
            self.cellpoints = newpoints #store number of cells (which is boundaries between populations)
            self.thetas     = newthetas

        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)  
        self.splits = np.array(self.splits)          
        self.totpts    = self.npts
        assert(self.totpts == self.npts)
        self.rule = rule
        self.cellid = np.arange(self.totpts, dtype = np.int32)
        self.inverseid =  pd.Series(data = np.arange(self.npts), index = self.cellid) #dict(zip(self.cellid, np.arange(self.npts)))
        
        return self.cellpops

    
    def addPopulation(self, popidx, rule, daughterType, npoppts, poppoints = None, popthetas = None):
        '''
            Append a population to the list of populations
        '''
        
        if poppoints is None:
            poppoints = np.zeros((0,2))
            popthetas = np.zeros(0)

        if (daughterType < 0) and (popidx == 0):
            daughterType = self.npops
        
        if   rule == 'stochastic':
            self.cellpops.append(cellulation(poppoints, popidx = popidx, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(popidx)))

        elif rule == 'size':
            self.cellpops.append(sizulation(poppoints, popidx = popidx, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(popidx)))            

        elif rule == 'size-fixrate': #constant rate, fate determination based on size
            self.cellpops.append(sizulationFR(poppoints, popidx = popidx, totpts = self.totpts, popthetas = popthetas, daughterType = daughterType, **self.popParams(popidx)))
                            

        elif rule == '0D':
            self.cellpops.append(cellulation0D(cellidx = np.arange(self.totpts, self.totpts + npoppts), popidx = popidx, totpts = self.totpts, 
                                                daughterType = daughterType, **self.popParams(popidx)))

        else:
            print('Invalid rule')
            assert(0)        
        
        self.totpts += npoppts
        
        return self.cellpops[-1]
    
    
    def addMutatedPopulation(self, from_popidx, pop_params = None, popname = None,
                             moveids = None, mut_ncells = 0, middle_cell = False):
        '''
            Add a mutated population to the list of populations
                By default the population is empty
            
            Attributes:
                from_popidx: population index to mutate
                
            To specify initilisation of random population specify:
                moveids: move cells by specific cell indices
                mut_ncells: randomly move a fixed number of cells to move from the original population
                middle_cell: if True, then a single middle cell is moved
        '''
        
        if pop_params is None:
            pop_params = {}
        
        #Set index for mutated population
        newpopidx = self.npops
        
        #Create mutant population with zero cells
        mut_pop = self.cellpops[from_popidx].mutate(newpopidx, pop_params, popname)      
   
        #Add the population to the list of populations
        self.cellpops = self.cellpops[:self.npops]
        self.cellpops.append(mut_pop)
        self.poppts = np.append(self.poppts, mut_pop.ncells) #add number of cells for population points
        self.npops += 1
        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0) 
        self.splits = np.array(self.splits)         
        self.updateSupraIndices(1, self.npops - 1) #Repairs references to indices for suprabasal layers        
        self.initSuprabasal(empty = False) #add suprabasal if needed
        
        for p in self.params:
            if type(p) is list:
                self.params[p].append(getattr(mut_pop, p))
                assert(len(self.params[p]) == self.npops)

        #Add cells to population
        if moveids is not None: #move specifed cells
            self.fixedMutate(from_popidx, newpopidx, moveids)
            
        if mut_ncells > 0: #randomly move a fixed number of cells
            self.randomMutate(from_popidx, newpopidx, mut_ncells)

        if middle_cell is True: #move a single middle cell
            middle_idx = np.array([self.splits[from_popidx] + np.argmin(np.sum((self.cellpoints[self.splits[from_popidx]:self.splits[from_popidx+1]] - self.L/2)**2,axis=1))])
            self.fixedMutate(from_popidx, newpopidx, self.cellid[middle_idx])
            
        #Ensures clones are based on initial cell indices
        self.reset(verbose = False)
        


    def updatePopulations(self, time):
        '''
            Update cell populations by allowing cells to divide and differentiate
        '''
        
        if self.npops > 1:
            newpoints = np.reshape(np.array([]), (0,2))
            newthetas = np.array([])
            newcellid = np.array([], dtype = np.int32)   
            
            for i in range(self.npops):
                self.cellpops[i].updatePoints(self.cellpoints[self.splits[i]:self.splits[i+1]])
                self.cellpops[i].updateThetas(self.thetas[self.splits[i]:self.splits[i+1]])                


            for i in range(self.npops):
                #update points, update thetas for cellulation class
                                    
                self.cellpops[i].updateTotpts(self.totpts)             
                 
                if time > 0:
                    self.cellpops[i].cellDecision(time, self) #cell-fate decisions
                
                self.totpts = self.cellpops[i].getTotpts() # get total points

            for i in range(self.npops):
                self.cellpops[i].updateTotpts(self.totpts)
                
            for i in range(self.npops):                
                #get points, get thetas 
                self.poppts[i]  = self.cellpops[i].ncells
                if self.poppts[i] > 0:              
                    newpoints = np.vstack([newpoints, self.cellpops[i].getPoints()])
                    newthetas = np.hstack([newthetas, self.cellpops[i].getThetas()])
                    newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                              
            self.npts = np.sum(self.poppts)
            self.cellpoints = newpoints
            self.thetas = newthetas
            self.cellid = newcellid
            self.splits = list(np.cumsum(self.poppts))
            self.splits.insert(0, 0) 
            self.splits = np.array(self.splits)           
            
        else: # n == 1
            #update points, update thetas    
            self.cellpops[0].updatePoints(self.cellpoints)
            self.cellpops[0].updateThetas(self.thetas)            
            
            self.cellpops[0].cellDecision(time, self) #cell fate decisions
            
            #get points, get thetas 
            self.cellpoints = self.cellpops[0].getPoints()
            self.thetas = self.cellpops[0].getThetas() 
            self.cellid = self.cellpops[0].getIndices()   
            self.poppts[0]  = self.cellpops[0].ncells            
            self.npts = int(self.cellpops[0].ncells)
            
        self.inverseid =  pd.Series(data = np.arange(self.npts), index = self.cellid) #dict(zip(self.cellid, np.arange(self.npts)))
        
        
        if self.suprabasal:
            self.updateSuprabasal(time)
        
        #print('Time: ', time, 'ncells:', self.poppts, 'nsb:', self.nsbcells, 'Totpts:', self.totpts)
        
        return self.cellpops


    def initSuprabasal(self, empty = True):
        '''
            Initialise the suprabasal layer
            
            Use empty = True, to default to an empty suprabasal layer
        '''

        if self.suprabasal is False:
            return #No suprabasal layer to initialise
        
        if empty:
            self.nsbcells = 0 #start with zero suprabasal cells
            self.sbas = cellulation0D(cellidx = set(), divtime = self.sbasshed, refract = self.sbasrefract,
                divprob = 0.0, totpts = self.totpts, assprob = 0.0, difprob = 0.0, 
                terprob = 0.0, daughterType = 10*self.npops, popidx = self.npops, motherType = 0,
                cv_div = self.sbas_cv)
        
        
        #Create the cellulation and add to end of cellpops
        self.cellpops = self.cellpops[:self.npops]
        self.cellpops.append(self.sbas)
        
    @property
    def adjmat(self):
        '''
            Returns the adjacency matrix of cells
        '''

        if self.time == self.adjmat_time and self._adjmat is not None:
            return self._adjmat
        
        if hasattr(self, 'jlengths'):
            self._adjmat = self.jlengths > 0
        else:
            self._adjmat = np.zeros((self.npts, self.npts), dtype = bool)
            ridge_in_pts = self.pbcidx[self.ridge_points[np.min(self.ridge_points,1) < self.npts,:]]
            self._adjmat[ridge_in_pts[:,0], ridge_in_pts[:,1]] = True
            self._adjmat[ridge_in_pts[:,1], ridge_in_pts[:,0]] = True
        self.adjmat_time = self.time

        return self._adjmat


        
    def updateSupraIndices(self, pops_added = 1, old_sup_idx = -1):
        '''
            Updates the index for the suprabasal layer
            
            This is required before adding new populations once 
                the suprabasal layer is initialised

            We use the the convention that daughterType = self.npops, 
                corresponds to cells moved to the suprabasal layer
        '''
        
        if old_sup_idx < 0:
            old_sup_idx = self.npops
        
        for p in range(self.npops):
            if self.cellpops[p].daughterType >= old_sup_idx:
                self.cellpops[p].daughterType += pops_added
        

        
    def updateSuprabasal(self, time):
        #Update the cellulation
         
        self.sbas.updateTotpts(self.totpts)
        
        if time > 0:
            self.sbas.cellDecision(time, self) #cell-fate decisions
        
        self.totpts = self.sbas.getTotpts() # get total points      
        self.nsbcells = self.sbas.ncells 
        
        assert(len(self.sbas.cellidx) == self.sbas.ncells)
        self.cellid = np.hstack([self.cellid, self.sbas.getIndices()]) 


    def addCell(self, popidx, d, pos, theta, anc = -1, parent = -1, sister = -1, dtime = None, delay = 0, drate = None):
        #dtime = None: use default division rate, 
        #dtime <= 0: daughter cannot divide 
        #dtime > 0: sets a time for the cell to divide

        #drate = None: use default division rate
        #drate  > 0: use adjusted division rate

        self.cellpops[popidx].addCell(d, pos, theta, self.time, anc, parent, sister, dtime, delay, drate)
        
        
    def removeCell(self, popidx, d):
        return self.cellpops[popidx].die(d, self.time)
        
    def termDivide(self, popidx, d):
        self.cellpops[popidx].updateTotpts(self.totpts)
        success = self.cellpops[popidx].termDivide(d, self, self.time)        
        self.totpts = self.cellpops[popidx].getTotpts()
        return success
        
    def divideFateChoice(self, popidx, d):
        self.cellpops[popidx].updateTotpts(self.totpts)
        success = self.cellpops[popidx].divideFateChoice(d, self, self.time)
        self.totpts = self.cellpops[popidx].getTotpts()
        return success



    def scheduleEvent(self, popidx, d, dtime = -1, drate = None):
        if dtime < 0: #use random time
            if drate <= 0:
                drate = self.cellpops[popidx].divtime
                if drate <= 0:
                    drate = self.cellpops[self.npops - 1].divtime
                    
            if drate <= 0: #If still no time, we do not schedule
                return 
            
            if self.time - self.cellpops[popidx].born[d] > self.cellpops[popidx].refract:
                dtime = self.cellpops[popidx].lifetime(drate = drate) + self.time
            else:
                dtime = self.cellpops[popidx].lifetime(drate = drate) + self.time + (self.cellpops[popidx].refract + self.cellpops[popidx].born[d] - self.time)
            #self.cellpops[popidx].born[d]
        
        self.cellpops[popidx].decTime.setdefault(next_floor(dtime), []).append(d)
    
    
    
    def removeRandomCell(self, popidx):
        self.cellpops[popidx].deleteRandom(self.time)
    
    
    def lineagePopulations(self): # get lineages
        if self.npops > 1 or self.suprabasal:
            descendents = [cp.desc for cp in self.cellpops]
        else:
            descendents = self.cellpops[0].desc
            
        if self.suprabasal:
            descendents.append(self.sbas.desc)
        
        return descendents
            
    def getAncestors(self):
        if self.npops > 1 or self.suprabasal:
            ancestors = [cp.ancs for cp in self.cellpops]
        else:
            ancestors = [self.cellpops[0].ancs]
            
            if self.suprabasal:
                ancestors.append(self.sbas.ancs)            
            
        return ancestors
    
    def getPopIndices(self, i):
        return self.cellid[self.splits[i]:self.splits[i+1]]
    
    def getDivisions(self):
        if self.npops > 1 or self.suprabasal:
            divs = [cp.dupTime for cp in self.cellpops]
        else:
            divs = self.cellpops[0].dupTime
            
        if self.suprabasal:
            divs.append(self.sbas.dupTime)                
            
        return divs
        
        
    def getDeaths(self):
        if self.npops > 1 or self.suprabasal:
            deaths = [cp.dieTime for cp in self.cellpops]
        else:
            deaths = self.cellpops[0].dieTime
            
        if self.suprabasal:
            deaths.append(self.sbas.dieTime)                  
            
        return deaths
     
    def getNeighbours(self, i, fullid = True): 
        ''' 
            fullid = True: takes in and returns the unique cell index
            fullid = False: takes in and returns the cell position index
        '''
        if fullid:
            return self.cellid[np.where(self.adjmat[self.inverseid[i],:] == 1)[0]]
        else:
            return np.where(self.adjmat[i,:] == 1)[0]
        

    def isNeighbour(self, i, j, fullid = True):
        #Checks if cells i and j are neighbours
        # i and j are the unique cell indices
        if fullid:
            return self.adjmat[self.inverseid[i], self.inverseid[j]] == 1
        else:
            return self.adjmat[i, j] == 1
    
    def subAdjmat(self, indices):
        #Returns the adjacency matrix for a subset of cells
        posidx = self.inverseid[indices]
        return self.adjmat[np.ix_(posidx, posidx)]


    def isAlive(self, p):
        return p in self.cellid
        
    def isInPop(self, p, popidx):
        return p in self.cellpops[popidx].getIndices()        
    
    
    def setVerbose(self, verbose):
        for j in range(len(self.cellpops)):
            self.cellpops[j].verbose = verbose

    
    def multify(self, x):
        if np.ndim(x) == 0 and self.npops > 1:
            x = np.ones(self.npops)*x
        elif type(x) is list:
            if self.npops > 1:
                x = np.array(x)
            else:
                x = x[0]
        return x
    
    
    def setup_sim_params(self, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0, p0 = 3.5):
        ''' 
            Sets up the simulation parameters, and multifies
        '''
        
        self.sim_params = self.default_sim_params.copy()
        
        if 'Dr' not in self.sim_params:
            self.sim_params['Dr'] = Dr
        
        if 'v0' not in self.sim_params:
            self.sim_params['v0'] = v0
            
        if 'mu' not in self.sim_params:
            self.sim_params['mu'] = mu
            
        if 'kA' not in self.sim_params:
            self.sim_params['kA'] = kA
            
        if 'kP' not in self.sim_params:
            self.sim_params['kP'] = kP
            
        if 'A0' not in self.sim_params:
            self.sim_params['A0'] = A0
            
        if 'p0' not in self.sim_params:
            self.sim_params['p0'] = p0
        
        #Ensure parameters are right dimension
        self.sim_params['Dr'] = self.multify(self.sim_params['Dr'])
        self.sim_params['v0'] = self.multify(self.sim_params['v0'])
        self.sim_params['mu'] = self.multify(self.sim_params['mu'])
        self.sim_params['kA'] = self.multify(self.sim_params['kA'])
        self.sim_params['kP'] = self.multify(self.sim_params['kP'])
        self.sim_params['A0'] = self.multify(self.sim_params['A0'])
        self.sim_params['p0'] = self.multify(self.sim_params['p0'])
        self.sim_params['P0'] = self.sim_params['p0']*np.sqrt(self.sim_params['A0']) #target perimeter
        
        return self.sim_params
    
    
    def shape_data(self):
        ''' 
            Calculates shape data from the current state of the tissue
        '''
        
        shape_tensor = np.zeros((self.npts,2,2))
        circularity  = np.zeros(self.npts)
        Saniso       = np.zeros(self.npts)
        Sorientation = np.zeros(self.npts)

        for p in range(self.npts):
            vertices = self.vertices[np.array(self.regions[self.point_region[p]]),:]
            centroid = np.mean(vertices,0)
            rel_vert = vertices - centroid
            shape_tensor[p,:,:] = np.mean(np.array([np.tensordot(rel_vert[v,:], rel_vert[v,:], axes = 0) for v in range(len(vertices))]),0)
            
            evals, evecs = np.linalg.eig(shape_tensor[p,:,:])
            idx = evals.argsort()#[::-1]   
            evals = evals[idx]
            evecs = evecs[:,idx]

            circularity[p] = evals[0]/evals[1]
            Saniso[p] = (evals[1] - evals[0])/np.sqrt(np.sum(np.square(evals)))
            Sorientation[p] = np.arctan2(*evecs[:,1]) % np.pi
            
        #Nematic order parameter, local and global
        edges = self.ridge_points[np.max(self.ridge_points,1) < self.npts]
        Qglobal = np.abs(np.mean(np.exp(2j*Sorientation)))
        rel_angle = Sorientation[edges[:,0]] - Sorientation[edges[:,1]] # + np.pi/2)%np.pi - np.pi/2
        Qlocal  = np.mean(np.cos(2*rel_angle))
    
        return {'Qlocal': Qlocal, 'Qglobal': Qglobal, 'circularity': circularity,
                'Saniso': Saniso, 'Sorientation': Sorientation, 
                'shape_tensor': shape_tensor, 'jlengths': self.junctionLengths(incpbc = False)}
    
    
    
    def save_shape_data(self, shapes = {}, save_shape = True):
        ''' 
            Obtains a dictionary with cell-shape data of the current state, 
                when requested to save
        '''
        
        shapes.setdefault('areas', []).append(self.areas[:self.npts].copy())
        #shapes.setdefault('jlengths', []).append(self.junctionLengths(incpbc = False).copy()) #uses way too much memory
        shapes.setdefault('perims', []).append(self.perim[:self.npts].copy())
        
        if save_shape:
            sd = self.shape_data()
            for k in sd.keys():
                shapes.setdefault(k,[]).append(sd[k].copy())
        
        return shapes    
        

    def sim(self, tsteps = 100, dt = 1e-2, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0, 
            p0 = 3.5, plot = False, skip = 10, verbose = False,
            save_shape = False, no_save = False,
            **kwargs):
        '''simulator to use without division and death
            Dr: rotational diffusion coefficient
            v0: magnitude of self-propulsion velocity of cells
            mu: mobility, inverse of frictional drag
            Ac, Pc: actual cell area/perimeter
            kA, kP: corresponding spring constants
            A0, P0: preferred cell area, perimeter
            plot: whether or not to plot at each time-step
            
            no_save: does not save time course information
        '''
        
        if skip <= 0:
            skip = 1
        
        #Set-up storage
        E = np.zeros(tsteps + 1)
        if no_save is False:
            cellpos = [self.cellpoints.copy()]            
            shapes = self.save_shape_data(shapes = {}, save_shape = save_shape)
        
        self.time = 0
        self.oldnpts = 0
        self.dt = dt
        
        print(self.npops, 'populations in sim')
        #Set the simulation parameters, and multify
        self.setup_sim_params(Dr, v0, mu, kA, kP, A0, p0)
        
        #Initial energy
        E[0] = self.E
                
        
        print('Starting sim with', self.npts, 'cells at', time.asctime())

        for i in range(1, tsteps + 1):
            self.PBCevolve(self.sim_params['Dr'], self.sim_params['v0'], self.sim_params['mu'], 
                           self.sim_params['kA'], self.sim_params['kP'], self.sim_params['A0'], self.sim_params['P0'], dt, plot)
            E[i] = self.E  
            self.time += 1
            
            if no_save is False and i % skip == 0: #saves data every 10 time poitnts
                cellpos.append(self.cellpoints.copy())
                shapes = self.save_shape_data(shapes, save_shape = save_shape)
                
                #areas.append(self.areas[:self.npts])
                #jlengths.append(self.junctionLengths(incpbc = False))
        
        assert(len(shapes['areas']) == len(cellpos))
        # Update population positions
        self.updatePopulations(-1) #negative time ensures no cell divisions
        self.end_sim()
        
        self.time = 0
        
        print('Ending sim with', self.npts, 'cells at', time.asctime())    
        if no_save:
            return {'E': E, 'skip': skip}
        else:
            return {'cellpos': cellpos, 'E': E, 'shapes': shapes, 'skip': skip}


    def end_sim(self, **kwargs):
        '''
            Ends the simulation, and updates the cell populations
            
            In future can add some options to reduce the amount we store
        '''
        
        self.addPBC(-1)
        self.updateVoronoi()

        for p in range(self.npops + int(self.suprabasal)):
            self.cellpops[p].end_sim(**kwargs)
        

        

    # simulations with division and death using cellulation class
    def simFull(self, tsteps = 100, dt = 1e-2, Dr = 0.0, v0 = 0.0, mu = 1.0, kA = 1.0, kP = 1.0, A0 = 1.0,
                p0 = 3.5, plot = False, skip = 10, maxCells = np.inf, minCells = 5, 
                verbose = False, save_shape = False, no_save = False, save_clonal = True, 
                **kwargs): 
        ''' 
            simulator to use with division and death 
                Dr: rotational diffusion coefficient
                v0: magnitude of self-propulsion velocity of cells
                mu: mobility, inverse of frictional drag
                Ac, Pc: actual cell area/perimeter
                kA, kP: corresponding spring constants
                A0, P0: preferred cell area, perimeter
                plot: whether or not to plot at each time-step
            
            no_save specifies whether or not to save time-course information
                save_clonal specifies whether to save clonal data
        '''        
             
        #Set-up storage
        E = np.zeros(tsteps + 1)
        if no_save is False:
            cellpos = [self.cellpoints.copy()]
            cellids = [self.cellid.copy()]
            shapes = self.save_shape_data(shapes = {}, save_shape = save_shape)
            
        
        if skip <= 0:
            skip = 1        
        
        ncells = np.zeros((tsteps + 1, self.npops))
        ncells[0,:] = self.poppts
        nsbcells = np.zeros(tsteps + 1)
        nsbcells[0] = self.nsbcells
        
        E[0] = self.E
        self.time = 0
        
        self.dt = dt
        self.oldnpts = 0             
        
        #Set the simulation parameters, and multify
        self.setup_sim_params(Dr, v0, mu, kA, kP, A0, p0)    
        
        #setting verbosity
        self.setVerbose(verbose)        
        
        print('Starting sim with', self.npts, 'cells at', time.asctime())
    
        
        for i in range(1, tsteps + 1):
            self.time += 1
            
            self.updatePopulations(i) #updates populations
            
            ncells[i, :] = self.poppts   
            nsbcells[i]  = self.nsbcells                     
            
            if self.npts >= maxCells or self.npts <= minCells:
                break        
        
            self.PBCevolve(self.sim_params['Dr'], self.sim_params['v0'], self.sim_params['mu'], 
                           self.sim_params['kA'], self.sim_params['kP'], self.sim_params['A0'], 
                           self.sim_params['P0'], dt, plot)
                    
            E[i] = self.E
            
            if no_save is False and i % skip == 0: #saves data every 10 time-points
                cellpos.append(self.cellpoints.copy())
                cellids.append(self.cellid.copy())
                assert(cellpos[-1].shape[0] == self.npts)
                assert(len(cellids[-1]) == (self.npts + self.nsbcells))
                
                shapes = self.save_shape_data(shapes, save_shape = save_shape)

                
        assert(len(shapes['areas']) == len(cellids))
        
        print('Ending sim with', self.npts, 'cells at', time.asctime()) 
          
        return_data = {'ncells': ncells.astype('int'), 'nsbcells': nsbcells.astype('int'), 'skip': skip, 'E': E, 'cellid': cellids,
                       'anc': self.getAncestors()}
        
        if no_save is False:
            return_data = {**return_data, 'cellpos': cellpos, 'shapes': shapes, 'desc': self.lineagePopulations(), 
            'dupTimes': self.getDivisions(), 'dieTimes': self.getDeaths()}
            
        if save_clonal is True:
            return_data = {**return_data}
        
        self.end_sim(**kwargs)
        
        return return_data    
    
    
    def runsuprabasal2(self, results, tsteps, intshed, sbasshed, skip = 10, npops = None, instantdelam2 = False):
        #Run with 2 suprabasal layers:
        #Run delamination to the suprabasal layer occuring via an intermediate state
        #Note that we don't relabel cells depending on which "suprabasal level" they're in
        
        
        #If no intermediate shedding provided, default to a single suprabasal layer
        if intshed == 0 and sbasshed > 0:        
            return self.rerunsuprabasal(results, tsteps, sbasshed, skip)
        
        if npops == None:
            npops = self.npops


        self.intshed  =  intshed #delamination rate for the intermediate cell-type
        self.sbasshed = sbasshed #shedding rate from the suprabasal layer
        
        newresults = results.copy()
        ncells = results['ncells'][:, :npops]
        cellids = results['cellid'].copy()
        
        
        #If already has intermediate layer:
        if len(self.cellpops) == npops + 2:
            self.sbas = self.cellpops[-2]
            
        
        #run the intermediate layer
        intlayer = self.sbas.copy() #intermediate layer
        intresults = intlayer.evolvecopy(tsteps, intshed, skip, instantdelam2 = instantdelam2, born = self.sbas.born)
        nintcells = intresults['ncells']
        self.totpts = intlayer.getTotpts()
        
        #Make the suprabasal layer
        born = {}
        for t in intlayer.dieTime.keys():
            for d in intlayer.dieTime[t]:
                born[d] = t
        
        #Run the suprabasal layer
        sbas = self.sbas
        sbasresults = sbas.evolvecopy(tsteps, sbasshed, skip, instantdelam2 =  False, born = born)     
        nsbcells = sbasresults['ncells']   
       
        
        #Check populations work
        assert(len(intresults['cellid'])  == len(cellids))
        assert(len(sbasresults['cellid']) == len(cellids))
        
        #Append all cellid's together
        for tt in range(0, int(tsteps/skip) + 1): 
            cellids[tt] = np.hstack([cellids[tt][:np.sum(ncells[tt*skip,:])], 
                             intresults['cellid'][tt],
                             sbasresults['cellid'][tt]]).astype('int64')
            #print(tt, sbasresults['cellid'][tt], nsbcells[tt*skip])
            #assert(len(sbasresults['cellid'][tt]) == nsbcells[tt*skip])
            assert(len(cellids[tt]) == np.sum(results['ncells'][tt*skip,:]) + nintcells[tt*skip] + nsbcells[tt*skip])

        #Append the populations     
        self.cellpops = self.cellpops[:npops]
        self.cellpops.append(intlayer)        
        self.cellpops.append(sbas)
        ncells = np.hstack([ncells, np.reshape(nintcells.copy(),(tsteps + 1,1))])        
        
        #Save the data
        newresults['nsbcells'] = nsbcells.copy()
        newresults['cellid']   = cellids
        newresults['ncells']   = ncells
        
        self.nsbcells = int(self.sbas.ncells)
        self.totpts = self.sbas.getTotpts()
        self.cellid = self.cellid[:self.npts]
        self.cellid = np.hstack([self.cellid, intlayer.getIndices(), self.sbas.getIndices()]) 
        #print(len(self.cellid), self.sbas.ncells + self.npts)
        #print(len(cellids[-1]), len(self.cellid), tt)        
        assert(len(self.cellid) == self.sbas.ncells + self.npts + intlayer.ncells)

        self.npops = npops + 1
        
        return newresults    
    
    

    def rerunsuprabasal(self, results, tsteps, divtime, skip = 10, 
                        shed_profile = None, tstart = 0, refract = 0):
        #evolve and extract results with a different decay (division) time
        
        self.sbasshed = divtime
        
        newresults = results.copy()
        ncells = results['ncells']

        if refract > 0:
            print('Running shedding with refract:', refract, ', shed', divtime)

        sbasresults = self.sbas.evolvecopy(tsteps, divtime, skip, shed_profile, tstart, refract = refract)     
        nsbcells = sbasresults['ncells']
        
        cellids = results['cellid'].copy()

        assert(len(sbasresults['cellid']) == len(cellids))
        assert(len(nsbcells) == len(results['ncells']))
        
        for tt in range(0, int(tsteps/skip) + 1): 
            cellids[tt] = np.hstack([cellids[tt][:np.sum(ncells[tt*skip,:])], 
                             sbasresults['cellid'][tt]]).astype('int64')
            #print(tt, sbasresults['cellid'][tt], nsbcells[tt*skip])
            #assert(len(sbasresults['cellid'][tt]) == nsbcells[tt*skip])
            assert(len(cellids[tt]) == np.sum(results['ncells'][tt*skip,:]) + nsbcells[tt*skip])

        
        newresults['nsbcells'] = nsbcells.copy()
        newresults['cellid']   = cellids
        
        self.nsbcells = int(self.sbas.ncells)
        self.totpts = self.sbas.getTotpts()
        self.cellid = self.cellid[:self.npts]
        self.cellid = np.hstack([self.cellid, self.sbas.getIndices()]) 
        #print(len(self.cellid), self.sbas.ncells + self.npts)
        #print(len(cellids[-1]), len(self.cellid), tt)        
        assert(len(self.cellid) == self.sbas.ncells + self.npts)
        
        return newresults
        
        
    def fixedMutate(self, frompop, topop, moveids):       
        for d in moveids:
            success = self.cellpops[frompop].differentiate(d, self.time, self, daughterType = topop)
            assert(success == True)

        #Repair cell populations

        self.totpts = self.cellpops[frompop].getTotpts() # get total points
        for i in range(self.npops):
            #update points, update thetas for cellulation class               
            self.cellpops[i].updateTotpts(self.totpts)        

        newpoints = np.reshape(np.array([]), (0,2))
        newthetas = np.array([])
        newcellid = np.array([], dtype = np.int32)

        for i in range(self.npops):                
            #get points, get thetas 
            self.poppts[i]  = self.cellpops[i].ncells
            if self.poppts[i] > 0:              
                newpoints = np.vstack([newpoints, self.cellpops[i].getPoints()])
                newthetas = np.hstack([newthetas, self.cellpops[i].getThetas()])
                newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                            
        self.npts = np.sum(self.poppts)
        self.cellpoints = newpoints
        self.thetas = newthetas
        self.cellid = newcellid
        self.splits = list(np.cumsum(self.poppts))
        self.splits.insert(0, 0)  
        self.splits = np.array(self.splits)
        self.inverseid =  pd.Series(data = np.arange(self.npts), index = self.cellid) #dict(zip(self.cellid, np.arange(self.npts)))
        

    def randomMutate(self, frompop, topop, numcells):
        '''
            moved (numcells) cells at random from one cell population (frompop) to another population (topop)
        '''

        #randomly generate indices
        assert(self.poppts[frompop] >= numcells)
        assert(len(self.cellpops[frompop].getIndices()) == self.poppts[frompop])
        chosen = self.cellpops[frompop].getIndices()[np.array(np.random.choice(range(self.poppts[frompop]), numcells, replace=False))]
        
        self.fixedMutate(frompop, topop, chosen)


    def plot(self): #Plot snapshot at end of tissue
        fig = voronoi_plot_2d(self, show_vertices = False, show_points = False)
        markers = ['o',  '^', 'o', '*', 'h', 'x']
        colors  = ['w', 'b', 'g', 'r', 'm', 'k']
        
        plt.plot(self.cellpops[0].getPoints()[:,0] % self.L, self.cellpops[0].getPoints()[:,1] % self.L, markers[0],  
                    label = self.cellpops[0].popname, color = colors[p], markersize = 0.000001)  
              
        for p in range(1,self.npops):
            plt.plot(self.cellpops[p].getPoints()[:,0] % self.L, self.cellpops[p].getPoints()[:,1] % self.L, markers[p],  
                     label = self.cellpops[p].popname, color = colors[p])

        if self.npops > 1:
            plt.legend()
            
        plt.axis('square')
        plt.xlim([0, self.L])
        plt.ylim([0, self.Ly])
        #plt.show()

        return fig

    def popnames(self):
        pn = []
        for p in range(self.npops):
            pn.append(self.cellpops[p].popname)

        return pn


    def init_clones(self):
        #Initialise clones
        self.clone


class tissue0D(tissue):

    def __init__(self, pop_idx = np.zeros([0]), divtime = 0, divprob = 0.0, assprob = 0.0, difprob = 0.0, terprob = 0.0, rule = '0D', 
                 sbasshed = 0, seed = None, poppts = None, sbasrefract = 0, sbas_cv = 1.0, **params):
        
        '''
            Removed default parameters (not in tissue class): points, thetas, L, PBC
        '''
        
        
        self.dim  = 0
        

        if seed is not None:
            seed = np.random.seed(seed)
            
        if poppts is not None:
            pop_idx = self.randomConfig(poppts)

        self.npts   = len(pop_idx)
        self.totpts = 0 #total number of cells in history of simulation
        
        self.time = 0
        
        #Suprabasal shedding time-scale < 0: no suprabasal layer
        if sbasshed < 0:
            self.suprabasal = False
            self.nsbcells = 0
        else:
            self.suprabasal  = True
            self.sbasshed    = sbasshed
            self.sbasrefract = sbasrefract
            self.sbas_cv     = sbas_cv
                
        self.splits = np.array([0, self.npts]) # default splits between points
        
        # Set-up populations
        self.makeParams(pop_idx, divtime, divprob, assprob, difprob, terprob, rule, **params)
        self.setupPopulations(pop_idx, divtime, rule, **params)


    def randomConfig(self, poppts):
        '''
            If input number of cells by population, initialises with randomly places cells:
        '''
        if np.ndim(poppts) == 0: #if input as a number
            poppts = [poppts]
        
        npts = np.sum(poppts)
        self.poppts = poppts
        splits = self.updateSplits()

        popidx = np.zeros(npts, dtype = np.int32)
        for i in range(len(poppts)):
            popidx[splits[i]:splits[i+1]] = i
        return popidx



    def sim(self, tsteps = 100, dt = 1e-2, skip = 10, maxCells = np.inf, minCells = 5, verbose = False, **kwargs):
        print('No spatial evolution. Use simFull')
        return



    def simFull(self, tsteps = 100, dt = 1e-2, skip = 10, maxCells = np.inf, minCells = 0, verbose = False, **kwargs):
            ''' simulator to use with division and death 
            '''        
                
            
            cellids = [] #[self.cellid.copy()]
            ncells = np.zeros((tsteps + 1, self.npops))
            ncells[0,:] = self.poppts
            
            nsbcells = np.zeros(tsteps + 1)
            nsbcells[0] = self.nsbcells
                        
            self.dt = dt
            self.oldnpts = 0   
            self.time = 0     
            
            #setting verbosity
            self.setVerbose(verbose)
                                            
            print('Starting sim with', self.npts, 'cells at', time.asctime())
            
            #Initialise time
            lasttime  = 0            
            
            while lasttime <= tsteps:
                ncells[lasttime:self.time, :] = self.poppts[None,:]  #update for all intermediate times
                nsbcells[lasttime:self.time]  = self.nsbcells  
                
                #Add cellids in between times
                for i in np.arange(int(np.ceil((lasttime)/skip))*skip, self.time, skip):
                    cellids.append(self.cellid.copy())                   
                
                self.updatePopulations(self.time) #updates populations
                lasttime = self.time
                self.time = np.min([self.jumpToNextEvent(), tsteps + 1])  
                    
                if self.npts >= maxCells or self.npts <= minCells:
                    break
                                
            self.time = tsteps
                                
            print('Ending sim with', self.npts, 'cells at', time.asctime()) 
            
            return {'ncells': ncells.astype('int'), 'desc': self.lineagePopulations(), 
            'anc': self.getAncestors(), 'cellid': cellids, 'dupTimes': self.getDivisions(),
            'dieTimes': self.getDeaths(), 'nsbcells': nsbcells.astype('int'), 'skip': skip}   



    def updatePopulations(self, time):
        '''
            Update cell populations by allowing cells to divide and differentiate
        '''
        
        if self.npops > 1:
            newcellid = np.array([], dtype = np.int32)   

            for i in range(self.npops):
                #update points, update thetas for cellulation class
                                    
                self.cellpops[i].updateTotpts(self.totpts)
                 
                if time > 0:
                    self.cellpops[i].cellDecision(time, self) #cell-fate decisions
                
                self.totpts = self.cellpops[i].getTotpts() # get total points

                
            for i in range(self.npops):                
                #get points, get thetas 
                self.poppts[i]  = self.cellpops[i].ncells
                if self.poppts[i] > 0:              
                    newcellid = np.hstack([newcellid, self.cellpops[i].getIndices()])                  
                              
            self.npts = np.sum(self.poppts)
            self.cellid = newcellid
            self.splits = list(np.cumsum(self.poppts))
            self.splits.insert(0, 0)  
            self.splits = np.array(self.splits)          
            
        else: # n == 1
            self.cellpops[0].cellDecision(time, self) #cell fate decisions
            
            #get points, get thetas 
            self.cellid = self.cellpops[0].getIndices()   
            self.poppts[0]  = self.cellpops[0].ncells            
            self.npts = int(self.cellpops[0].ncells)
            
        self.inverseid =  pd.Series(data = np.arange(self.npts), index = self.cellid) #dict(zip(self.cellid, np.arange(self.npts)))
        
        if self.suprabasal:
            self.updateSuprabasal(time)
        
        #print('Time: ', time, 'ncells:', self.poppts, 'nsb:', self.nsbcells, 'Totpts:', self.totpts)
        
        return self.cellpops
    
    
    
    def jumpToNextEvent(self):
        if self.suprabasal is True:
            npops = self.npops + 1
        else:
            npops = self.npops
       
        newtime = np.min([self.cellpops[p].nextEventTime() for p in range(npops)])
        
        if newtime == np.inf:
            newtime = 2**31 - 1 #set to maximum integer value

        if newtime <= self.time:
            print(newtime, self.time, 'ERROR: trying to jump the past')
            assert(0)

        return int(newtime)
    
    
    def reset(self, full = False, verbose = True):
        '''reset's populations retaining final point locations, thetas, 
            cells of each population type and

            Note: the dead cell will be left in decision time, unless a full reset is used.
        '''
        if full == True:
            oldtime = 0
        else:
            oldtime = self.time

        if verbose:
            print('Resetting time and division times')        
        
        for i in range(self.npops):
            self.cellpops[i].reset(oldtime)
            
        self.cellid = self.cellid[:self.npts]
        
        if self.suprabasal:
            self.sbas.reset(oldtime)
            #self.initSuprabasal()
            self.sbas.clear()
            self.nsbcells = 0
                   
        self.time = 0      


def save_tissue(results, filename, tiss = None):
    '''
        Note that BornDistribution object can be problematic to pickle
            due to NumericalInversePolynomial so we must remove before saving.
    '''

    if tiss is None:
        if 'tiss' in results:
            tiss = results['tiss']

    if tiss is not None:
        #Ensure tissue is saveable
        for p in range(len(tiss.cellpops)):
            tiss.cellpops[p].div1_dist = None

        results['tiss'] = tiss
    
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)    
