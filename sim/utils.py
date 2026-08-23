import numpy as np
#import matplotlib.pyplot as plt
#from scipy.spatial import Voronoi, voronoi_plot_2d
from numba import jit
#import time

#functions 

# Note vertices, ridge_points, ridge_vertices, point_region are defined as in "scipy.spatial.Voronoi"

# Numerical method for SPV model follows Appendix A of Bi et. al 2016 PRX (DOI: 10.1103/PhysRevX.6.021011)

@jit(nopython=True)
def vertexPos(ri, rj, rk):  #Calculates vertex position given 3 point positions
    #assumes ri, rj, rk have two components 
    rij = ri - rj
    rik = ri - rk
    rjk = rj - rk
    
    a = np.dot(rjk, rjk)*np.dot(rij,  rik)
    b = np.dot(rik, rik)*np.dot(-rij, rjk)
    c = np.dot(rij, rij)*np.dot(rik,  rjk)    
    D = 2*(rij[0]*rjk[1] - rjk[0]*rij[1])**2
    return (a*ri + b*rj + c*rk)/D

def addBoundary(points, nb, L, dim = 2, shift = False, Ly = None): 
    #Add boundary layer of cells (useful for non-periodic boundary conditions)
    #nb: boundary cells per edge, shift is the edge of the box
    
    if Ly is None:
        Ly = L    
    
    newpts = np.zeros([nb*4, dim])
    newpts[0:nb, 0] = L/nb*(np.arange(nb) + 1)
    newpts[nb:2*nb, 1] = Ly/nb*np.arange(nb)
    newpts[2*nb:3*nb, :2] = np.vstack([L/nb*np.arange(nb), Ly*np.ones(nb)]).transpose()
    newpts[3*nb:, :2] = np.vstack([L*np.ones(nb), Ly/nb*(np.arange(nb) + 1)]).transpose() 
    if shift:
        newpts[:, 0] -=  L/2 #shift
        newpts[:, 1] -= Ly/2 #shift
        
    if len(points) == 0:
        points = newpts    
    elif not np.shape(points) == dim:
        points = np.vstack([points, newpts])
    else:
        print('Failed to add boundary')
    return points
    

def randomConfig(npts, L, boundPoints = 0, seed = None, Ly = None):
    # Points generated from uniform distribution
    # Option to add a boundary layer of cells
    
    if Ly is None:
        Ly = L
    
    if seed is not None:
        np.random.seed(seed) 
         
    points = np.random.uniform(0, 1, [npts,2])*np.array([L,Ly])[None,:]
    
    if boundPoints > 0:
        points = addBoundary(points, boundPoints, L, Ly)
        npts += 4*boundPoints
    
    return points

def generateLattice(L, N = None, z = 4, Ly = None, noise = 0.0):
    # generates a lattices of N^2 points on an LxL grid
    #z = coordination number: 4 (square lattice), 6 hexagonal lattice
    
    if N is None:
        N = L
    
    if Ly is None:
        Ly = L
        
    if z == 4:
        points = np.zeros([N**2,2])
        for i in range(N):
            points[(i*N):(i + 1)*N, 0] = np.arange(N)*L/N
            points[(i*N):(i + 1)*N, 1] = i*Ly/N
    
    elif z == 6:
        points = np.zeros([N**2,2])
        for i in range(N):
            points[(i*N):(i + 1)*N, 0] = np.arange(N)*L/N + L/N/2*(i%2)
            points[(i*N):(i + 1)*N, 1] = i*L/N

        points = (points + noise*np.random.rand(N**2,2)) % L

    return points

@jit(nopython=True)
def PolyArea(verts, region): #signed area of polygon, region must be numpy array
    #output is positive if orientated anti-clockwise
    inc = region > -1 #points to include
    return 0.5*(np.sum(inc[1:]*inc[:-1]*(verts[1:,0] - verts[:-1,0])*(verts[1:,1] + verts[:-1,1])) + inc[0]*inc[-1]*(verts[0,0] - verts[-1,0])*(verts[0,1] + verts[-1,1]))


@jit(nopython=True)
def PolyPerim(edgeLengths, region): #perimeter of cell figure
    return np.sum(np.array([edgeLengths[region[i], region[i - 1]]*(region[i] > -1)*(region[i - 1] > -1) for i in range(len(region))]))

'''
@jit(nopython=True)
def PolyPerim(edgeLengths):
    return np.sum(edgeLengths, 1)
'''

@jit(nopython=True)
def dhdri_num(ri, rj, rk, mu, h = None, dx = 1e-8): #numerical solution
    #for vertex defined by ri, rj, rk computes dh_l/dr_i,nu
    if h is None:
        h = vertexPos(ri, rj, rk)
    vec = np.zeros(2)
    vec[mu] = dx
    return (vertexPos(ri + vec, rj, rk) - h)/dx
 
@jit(nopython=True)
def dhdri(ri, rj, rk, mu): #analytical solution
    #for vertex defined by ri, rj, rk computes dh_l/dr_i,mu
    rij = ri - rj
    rik = ri - rk
    rjk = rj - rk  
    
    drr = (1-mu)*np.array([1,0]) + mu*np.array([0,1]) #d/dr_i,nu(r_i)
    D = 2*(rij[0]*rjk[1] - rjk[0]*rij[1])**2 #2*(rij x rjk)^2
    dD = 4*rjk[(1 - mu)]*np.dot(rij,rjk) #dD/dr_i,nu
    da = np.dot(rjk, rjk)*(rik[mu] + rij[mu] - np.dot(rij, rik)*np.dot(rjk, rjk)*dD/D)/D #dalpha/dr_i,nu
    db = (np.dot(rik, rjk)*rij[mu] + np.dot(rik, rik)*(-rjk[mu] + np.dot(rij, rjk)*dD/D))/D   #dbeta/dr_i,nu
    dY = (np.dot(rij, rjk)*rij[mu] + np.dot(rik, rik)*(-rjk[mu] + np.dot(rij, rjk)*dD/D))/D   #dgamma/dr_i,nu
    a = np.dot(rjk, rjk)*np.dot(rij,  rik)
    return da*ri + db*rj + dY*rk + a*drr

#@jit(nopython = True)
def all_dhdr(points, tri, npts, nv, vertices):
    # computes dh_l/dr_i,mu for all points required in energy functional
    # there is some redundancy for points in boundary region
    dhdr = np.zeros((nv + 1, npts + 1, 2, 2), dtype = np.float64) #vertex, point, nu, mu #extra vertex ensures boundary edges map to zero
    
    for i in range(nv):
        t = tri[i, :]
        
        if t[0] < npts:
            dhdr[i, t[0], :, 0] = dhdri_num(points[t[0],:], points[t[1],:], points[t[2],:], 0, vertices[i,:])
            dhdr[i, t[0], :, 1] = dhdri_num(points[t[0],:], points[t[1],:], points[t[2],:], 1, vertices[i,:])   
            
        if t[1] < npts:     
            dhdr[i, t[1], :, 0] = dhdri_num(points[t[1],:], points[t[0],:], points[t[2],:], 0, vertices[i,:])
            dhdr[i, t[1], :, 1] = dhdri_num(points[t[1],:], points[t[0],:], points[t[2],:], 1, vertices[i,:])        
            
        if t[2] < npts:
            dhdr[i, t[2], :, 0] = dhdri_num(points[t[2],:], points[t[0],:], points[t[1],:], 0, vertices[i,:])
            dhdr[i, t[2], :, 1] = dhdri_num(points[t[2],:], points[t[0],:], points[t[1],:], 1, vertices[i,:])
            
        #if np.sum(np.isnan(dhdr[i, :,:,:])) > 0:
        #    print(i, t, points[t[0],:], points[t[1],:], points[t[2],:], 0, vertices[i,:])
        
    return dhdr



def all_dEdh(vertices, regions, point_region, points, Ac, Pc, KA, KP, A0, P0, npts, interior, pbcidx, npbcpts, ridge_points, pop_splits = []):
    # computes dE_j/dh_a,nu for all points required in energy functional
    # there is some redundancy for points in boundary region
    
    nv   = len(vertices)
    dEdh = np.zeros((npbcpts, nv, 2), dtype = np.float64)
    
    out_touch_in = touching_inner(ridge_points, npts)
        

    if len(pop_splits) >= 3:  #more than one population 
        #Loop over inner cells 
        for p in range(len(pop_splits) - 1):
            for i in range(pop_splits[p], pop_splits[p+1]):#np.array(interior[pop_splits[p]:pop_splits[p+1]]).nonzero()[0]:
                assert(interior[i] == True)
                assert(pbcidx[i] == i)
                vertIdx = np.array(regions[point_region[i]], dtype = np.int32)
                dEdh[i, vertIdx, :] = dEdhi(vertices[vertIdx, :], Ac[pbcidx[i]], Pc[pbcidx[i]], KA[p], KP[p], A0[p], P0[p], len(vertIdx)) 
    else:        
        for i in np.arange(npts):
            assert(interior[i] == True)            
            vertIdx = np.array(regions[point_region[i]], dtype = np.int32)
            dEdh[i, vertIdx, :] = dEdhi(vertices[vertIdx, :], Ac[pbcidx[i]], Pc[pbcidx[i]], KA, KP, A0, P0, len(vertIdx)) 
                
    #Loop over cells which touch inner cells
    ''' 
        Note: if we encounter future issues, a working solution is to loop over only 
            vertices where outer cells touch inner cells. These can be mapped to vertices
            of the corresponding inner cell inside the box. This is looping over
            ridge_vertices, but drawing the map from the outer to inner cell might
            be difficult to do in an efficient way.
    '''
    for i in out_touch_in:
        #assert(interior[i] == True)
        vertIdxOut = np.array(regions[point_region[i]], dtype = np.int32)
        vertsInBox = np.array(regions[point_region[pbcidx[i]]], dtype = np.int32)
        #assert len(vertIdxOut) == len(vertsInBox) , (i, pbcidx[i], vertIdxOut, vertsInBox)
        
        if len(vertIdxOut) == len(vertsInBox):
            vtrans = find_vertex_mapping(vertices[vertIdxOut,:], vertices[vertsInBox,:], points[i,:], points[pbcidx[i],:])
        else:
            vtrans = None
            
        if vtrans is not None:
            dEdh[i, vertIdxOut, :] = dEdh[pbcidx[i], vertsInBox[vtrans], :]
        else: 
            '''
                Find the intersecting vertices, ignore those that are not common
                This occurs when the outer vertices of outer cells are different from the
                equivalent vertices of the inner cells
            '''
            #print('Warning: vertices of inner and outer cells do not match', vertices[vertIdxOut,:], vertices[vertsInBox,:])
            vout, vin = matching_vertex_pairs(vertices[vertIdxOut,:], vertices[vertsInBox,:], points[i,:], points[pbcidx[i],:])
            dEdh[i, vertIdxOut[vout], :] = dEdh[pbcidx[i], vertsInBox[vin], :]
        
        
        '''
        #Testing that we have done the right thing
        p = pop_by_idx(pbcidx[i], pop_splits)
        
        #The key thing here is that Ac encodes a sign, 
        # #so we need to ensure that the sign is preserved upon transformation
        sign = 1 - 2*cyclic_reversed(vtrans)
        check = dEdhi(vertices[vertIdxOut, :], Ac[pbcidx[i]]*sign, Pc[pbcidx[i]], KA[p], KP[p], A0[p], P0[p], len(vertIdxOut))
        """
        if np.allclose(dEdh[i, vertIdxOut, :], check) == False:
            print(i, pbcidx[i], p, vertIdxOut, vertsInBox[vtrans], check, dEdh[i, vertIdxOut, :])
            print(vertices[vertIdxOut,:] - points[i,:], vertices[vertsInBox[vtrans],:] - points[pbcidx[i],:])
        """
        assert(np.allclose(dEdh[i, vertIdxOut, :], check))
        '''
        

    return dEdh



"""
def all_dEdh(vertices, regions, point_region, Ac, Pc, KA, KP, A0, P0, npts, interior, pbcidx):
    # computes dE_j/dh_a,nu for all points required in energy functional
    # there is some redundancy for points in boundary region
    
    nv   = len(vertices)
    dEdh = np.zeros((npts, nv, 2), dtype = np.float64)
    
    for i in np.array(interior[:npts]).nonzero()[0]:
        vertIdx = np.array(regions[point_region[i]], dtype = np.int32)
        dEdh[i, vertIdx, :] = dEdhi(vertices[vertIdx, :], Ac[pbcidx[i]], Pc[pbcidx[i]], KA, KP, A0, P0, len(vertIdx)) 
        
        #if np.sum(np.isnan(dEdh[i,:,:])) > 0:
        #    print(i, t, vertIdx)        
    
    
    return dEdh
"""

@jit(nopython = True)
def pop_by_idx(i, pop_splits):
    '''
        Given splits obtain the population of the i-th cell (by determining population)
            returns -1, otherwise
            
        Assumes pop_splits is a numpy array
    '''
    popidx = np.argmax(pop_splits > i) - 1
    assert(popidx != -1)
    return popidx


def pop_by_indices(idxs, pop_splits):
    '''
        Given list of indices, returns the populations of each index

        i.e. vectorised version of pop_by_idx
    '''
    return np.argmax(np.array(pop_splits)[:,None] > idxs[None,:],0) - 1


#@jit(nopython = True)
def touching_inner(ridge_points, npts):
    '''
        Outer cells which touch inner cells
    '''
    srp = np.sort(ridge_points, 1) #sorted ridge points
    return np.unique(srp[(srp[:,0] < npts) & (srp[:,1] >= npts), 1])

    
    

def all_dEdh1(vertices, regions, point_region, Ac, Pc, KA, KP, A0, P0, npts, interior, pbcidx, pop_splits = [], ridge_points = None, ridge_vertices = None, npbcpts = 0):
    '''
        Computes dE_j/dh_a,nu for all points required in energy functional
        there is some redundancy for points in boundary region    
    '''
    
    
    
    nv   = len(vertices)
    dEdh = np.zeros((npbcpts, nv, 2), dtype = np.float64)    
    
    for k in range(len(ridge_points)): #dEj/dri_mu
        if np.min(ridge_points[k, :]) < npts: #if one of the points is inside box
            i = ridge_points[k, 0] 
            j = ridge_points[k, 1]
            v = ridge_vertices[k, 0]
            w = ridge_vertices[k, 1]
            
            pi = pop_by_idx(pbcidx[i], pop_splits)
            pj = pop_by_idx(pbcidx[i], pop_splits)
            
            dEdh[i,v,:] += 2*KA*(np.abs(Ac[pbcidx[i]]) - A0[pi])
            dEdh[i,v,:] += 2*KP*(Pc[pbcidx[i]] - P0[pi])*(vertices[v,:] - vertices[w,:])/np.linalg.norm(vertices[v,:] - vertices[w,:])
            #verts[v,:] = 2*KA*(abs(Ac) - A0)*dAdh(vertices, v, np.sign(Ac)) + 2*KP*(Pc - P0)*dPdh(vertices, v)
            
    
    return


@jit(nopython = True)
def computeForces(ridge_points, ridge_vertices, point_region, dEdh, dhdr, npts, interior, pbcidx):
    # Computes forces on all points (eq. A7, Bi et. al 2016)
    
    forces = np.zeros((npts, 2), dtype = np.float64)
    
    #loops over ridges connecting points
    for k in range(len(ridge_points)): #dEj/dri_mu
        i = ridge_points[k, 0] 
        j = ridge_points[k, 1]
        v = ridge_vertices[k, 0]
        w = ridge_vertices[k, 1]
  
        if interior[i] and interior[j]:
            if i < npts:
                jj = j #pbcidx[j] #map onto points in the main box, does not work because vertices not preserved
                #if (np.sum(np.abs(dEdh[jj,v,0]*dhdr[v,i,0,:])) == 0) or (np.sum(np.abs(dEdh[jj,v,1]*dhdr[v,i,1,:])) == 0):
                #    print(k, i, j, v, w, dEdh[jj,v,:], dhdr[v,i,0,:], dhdr[v,i,1,:])
                #    assert((np.sum(np.abs(dEdh[jj,v,0]*dhdr[v,i,0,:])) != 0) and (np.sum(np.abs(dEdh[jj,v,1]*dhdr[v,i,1,:])) != 0))
                #assert((np.sum(np.abs(dEdh[jj,w,0]*dhdr[w,i,0,:])) != 0) and (np.sum(np.abs(dEdh[jj,w,1]*dhdr[w,i,1,:])) != 0))                
                forces[i, :] -= dEdh[jj,v,0]*dhdr[v,i,0,:] + dEdh[jj,v,1]*dhdr[v,i,1,:]
                forces[i, :] -= dEdh[jj,w,0]*dhdr[w,i,0,:] + dEdh[jj,w,1]*dhdr[w,i,1,:]

            if j < npts:                
                ii = i #pbcidx[i] #map onto points in the main box, does not work because vertices not preserved
                #if (np.sum(np.abs(dEdh[ii,v,0]*dhdr[v,j,0,:])) == 0) or (np.sum(np.abs(dEdh[ii,v,1]*dhdr[v,j,1,:])) == 0):
                #    print(k, i, j, v, w, dEdh[ii,v,:], dhdr[v,j,0,:], dhdr[v,j,1,:])
                #
                #assert((np.sum(np.abs(dEdh[ii,v,0]*dhdr[v,j,0,:])) != 0) and (np.sum(np.abs(dEdh[ii,v,1]*dhdr[v,j,1,:])) != 0))
                #assert((np.sum(np.abs(dEdh[ii,w,0]*dhdr[w,j,0,:])) != 0) and (np.sum(np.abs(dEdh[ii,w,1]*dhdr[w,j,1,:])) != 0))
                forces[j, :] -= dEdh[ii,v,0]*dhdr[v,j,0,:] + dEdh[ii,v,1]*dhdr[v,j,1,:]
                forces[j, :] -= dEdh[ii,w,0]*dhdr[w,j,0,:] + dEdh[ii,w,1]*dhdr[w,j,1,:] 
            
        if i < npts and interior[i]: #self-energy term
            forces[i, :] -= 0.5*(dEdh[i,v,0]*dhdr[v,i,0,:] + dEdh[i,v,1]*dhdr[v,i,1,:])
            forces[i, :] -= 0.5*(dEdh[i,w,0]*dhdr[w,i,0,:] + dEdh[i,w,1]*dhdr[w,i,1,:])

        if j < npts and interior[j]: #self-energy term
            forces[j, :] -= 0.5*(dEdh[j,v,0]*dhdr[v,j,0,:] + dEdh[j,v,1]*dhdr[v,j,1,:])
            forces[j, :] -= 0.5*(dEdh[j,w,0]*dhdr[w,j,0,:] + dEdh[j,w,1]*dhdr[w,j,1,:])          
    '''
    for i in range(npts): #alternate method for self-energy term, quicker for no numba
        vertIdx = regions[point_region[i]]
        if interior[i]:
            forces[i,:] -= np.sum(np.array([dEdh[i,v,0]*dhdr[v,i,0,:] + dEdh[i,v,1]*dhdr[v,i,1,:] for v in vertIdx]), axis = 0) #check axis
    '''

    
    return forces


@jit(nopython=True)
def ridgeLengths(vertices, ridge_vertices, ridge_points, npbcpts):
        '''
            Stores the lengths of edges by vertex indices
        '''
        
        #use ridge_vertices as np.array
        dist = norm2(vertices[ridge_vertices[:,0],:] - vertices[ridge_vertices[:,1],:])
        #dist = np.linalg.norm(vertices[ridge_vertices[:,0],:] - vertices[ridge_vertices[:,1],:], axis = 1)
        rl = np.zeros((npbcpts, npbcpts), dtype = np.float64)
        for i in range(len(dist)):
            rl[ridge_points[i,0], ridge_points[i,1]] = dist[i]
        rl += rl.transpose()
        #rl[ridge_vertices[:,1], ridge_vertices[:,0]] = dist
        return rl      

@jit(nopython=True)
def norm2(x):
    '''
        Norm of 2D vectors
    '''
    return np.sqrt(x[:,0]*x[:,0] + x[:,1]*x[:,1])

@jit(nopython=True)
def dAdh(vertices, v, sgn): #change in area, given chance in vertex, 
    # v is index of vertex, sgn = 1  / -1 for anticlockwise/clockwise ordering
    return np.flip(sgn*(vertices[v - 1] - vertices[(v + 1) % len(vertices)])/2)*np.array([1, -1]) #indexing is opposite direction to before

def dAdh_num(vertices, v, nu): #same as dA/dh but numerical
    dv = vertices.copy()
    dx = 1e-8
    dv[v, nu] += dx
    #print(PolyArea(vertices, np.zeros(len(vertices))), PolyArea(vertices, np.zeros(len(vertices))))
    return (abs(PolyArea(dv, np.zeros(len(vertices)))) - abs(PolyArea(vertices, np.zeros(len(vertices)))))/dx
    
@jit(nopython=True)
def dPdh(vertices, v): #change in area, given change in vertex
    #v is index of vertex
    return (vertices[v] - vertices[v - 1])/np.linalg.norm(vertices[v] - vertices[v - 1]) + (vertices[v] - vertices[(v + 1) % len(vertices)])/np.linalg.norm(vertices[v] - vertices[(v + 1)  % len(vertices)])

@jit(nopython=True)
def dPdh_num(vertices, v, nu): #same as dP/dh but numerical
    dv = vertices.copy()
    dx = 1e-8
    dv[v, nu] += dx
    P1 = np.sum(np.array([np.linalg.norm(vertices[i,:] - vertices[i - 1,:]) for i in range(len(vertices))]))
    P2 = np.sum(np.array([np.linalg.norm(dv[i,:] - dv[i - 1,:]) for i in range(len(dv))]))
    return (abs(P2) - abs(P1))/dx

@jit(nopython=True)
def dEdhi(vertices, Ac, Pc, KA, KP, A0, P0, nv):
    # computes dE_j/dh_a,nu given np.array of vertices corresponding to a point
    #Ac, Pc: actual cell area/perimeter
    #kA, kP: corresponding spring constants
    #A0, P0: preferred cell area, perimeter
    #nv: number of vertices for this point
    
    verts = np.zeros((nv, 2), dtype = np.float64)
    for v in range(nv):
        verts[v,:] = 2*KA*(abs(Ac) - A0)*dAdh(vertices, v, np.sign(Ac)) + 2*KP*(Pc - P0)*dPdh(vertices, v)
    return verts


def makeRings(ridge_points, npts):
    # given ridge-points, constructs rings of points around a vertex
    # not used in current implementation
    
    ringlist = [[] for i in range(npts)]
    #order doesn't matter
    for rp in ridge_points:
        ringlist[rp[0]].append(rp[1])
        ringlist[rp[1]].append(rp[0])
    return ringlist

def orderedRegions(regions, point_region):
    # re-orders: regions (from Voronoi object) to be in the same order as points
    newregion = []
    for r in point_region:
        newregion.append(regions[r])
    return newregion

def partitionTypes(class_idx, nclasses = None):
    '''
        Given list of class indices class_idx (ranging of from 0 to nclasses - 1)
            partition x into types, returns list of indices corresponding to each type
    '''
    
    if nclasses == None:
        nclasses = max(class_idx) + 1
    
    partitions = [[] for i in range(nclasses)]
    
    for i in range(len(class_idx)):
        partitions[class_idx[i]].append(i)
        
    return partitions

@jit(nopython=True)
def hill(x, n = 2):
    x_n = np.abs(x)**n
    return x_n/(1 + x_n)*(x > 0)


@jit(nopython=True)
def relu(x, m = 1, x0 = 0):
    return m*(x-x0)*(x > x0)
    
    
@jit(nopython=True)
def nablaf(f, h): #discrete Laplacian of f using five-point central difference, periodic boundary conditions
    #f returns to LxL matrix for distribution function, h: spacing
    #internally
    nf = -4*f.copy()
    nf[:,:-1] += f[:,1:]
    nf[:,1:]  += f[:,:-1]
    nf[:-1,:] += f[1:,:]
    nf[1:,:]  += f[:-1,:]
    
    # implement periodic boundary conditions
    nf[0,:]  += f[-1,:]
    nf[-1,:] += f[0,:]    
    nf[:,0]  += f[:,-1]
    nf[:,-1] += f[:,0]
    
    return 1/h**2*nf
    
    
    
@jit(nopython=True)
def meshgrid2dflat(minx, maxx, miny, maxy):
    szx = maxx - minx
    szy = maxy - miny
    
    xx = np.ones(shape=(szx, szy), dtype = 'int64')*minx
    yy = np.ones(shape=(szx, szy), dtype = 'int64')*miny
    
    for i in range(szx):
        for j in range(szy):
            xx[i,j] += i  # change to x[k] if indexing xy
            yy[i,j] += j  # change to y[j] if indexing xy
                
    return xx.flatten(), yy.flatten()


def rasteriseCells(idxs, vertices, regions, point_region, perArea = True, dx = 0.1, L = 15, Ly = None):
    if Ly is None:
        Ly = L
    #If the cell is too small the raster can give 0
    nx = int(L/dx)
    ny = int(Ly/dx)
    
    allgrid = np.zeros((nx,ny), dtype = 'float64')
    for p in idxs:
        verts   = np.array(vertices[regions[point_region[p]],:]/dx, dtype = 'float')        
        allgrid = updateRaster(allgrid, verts, nx, ny, dx, perArea)
            
    return allgrid


@jit(nopython=True)
def meshgrid2dflat(minx, maxx, miny, maxy):
    szx = maxx - minx
    szy = maxy - miny
    
    xx = np.ones(shape=(szx, szy), dtype = 'int64')*minx
    yy = np.ones(shape=(szx, szy), dtype = 'int64')*miny
    
    for i in range(szx):
        for j in range(szy):
            xx[i,j] += i  # change to x[k] if indexing xy
            yy[i,j] += j  # change to y[j] if indexing xy
                
    return xx.flatten(), yy.flatten()


@jit(nopython=True)
def updateRaster(allgrid, verts, nx, ny, dx, perArea = True): 
    #perArea = True, consumption per area
    #triangulate convex polygon by using vertex 0 as reference, loop over other pairs

    minx = int(np.floor(np.min(verts[:,0])))
    maxx = int(np.floor(np.max(verts[:,0]))) + 1
    miny = int(np.floor(np.min(verts[:,1])))
    maxy = int(np.floor(np.max(verts[:,1]))) + 1     
    
    xv, yv = meshgrid2dflat(minx, maxx, miny, maxy)
    contcell = np.zeros((maxx - minx)*(maxy - miny), dtype = 'float64')
    
    for v in range(1,len(verts)-1, 1):
        a = 0
        b = v
        c = v + 1

        as_x = xv - verts[a,0]
        as_y = yv - verts[a,1]

        s_ab = ((verts[b, 0] - verts[a, 0])*as_y - (verts[b, 1] - verts[a, 1])*as_x) > 0

        
        contcell += 1 - (((((verts[c,0] - verts[a,0])*as_y - (verts[c,1] - verts[a,1])*as_x) > 0) == s_ab) | 
            ((((verts[c,0] - verts[b,0])*(yv - verts[b,1]) - (verts[c,1] - verts[b,1])*(xv - verts[b,0])) > 0) != s_ab))
  
    if perArea == False and np.sum(contcell) > 0: #second check to ensure doesn't normalise to 0
        enclosed = np.sum(contcell > 0)*dx**2
        contcell = (contcell > 0)/enclosed  
        
    for i in range(len(xv)):
        allgrid[xv[i] % nx, yv[i] % ny] += contcell[i]

    return allgrid


def addCircle(gridsize, centre, area, L, gridsizey = None):
    if gridsizey is None:
        gridsizey = gridsize
    
    source = np.zeros((gridsize, gridsizey))
    
    dx = L/gridsize
    c  = np.round(np.array(centre)/dx).astype('int')
    r  = np.round(np.sqrt(area/np.pi)/dx).astype('int')
    
    for x in range(-r, r+1):
        Y = int((r**2 - x**2)**0.5) # bound for y given x
        for y in range(-Y, Y+1):
            source[(c[0] + x) % gridsize, (c[1] + y) % gridsizey] = 1
    
    return source


def makeSource(gridsize, centres, L, onsite = 1.0, offsite = 0.0, area = 1.0, gridsizey = None):
    if gridsizey is None:
        gridsizey = gridsize
            
    centres = np.array(centres)
    if np.ndim(centres) == 1:
        centres = np.reshape(centres, [1,2])
    
    source = np.ones((gridsize, gridsizey))*offsite
    npts = np.shape(centres)[0]

    for p in range(npts):
        source += (onsite - offsite)*addCircle(gridsize, centres[p,:], area, L, gridsizey)
        
    return source


@jit(nopython=True)
def find_vertex_mapping(vertices1, vertices2, points1, points2):
    ''' 
        Given a vertices array (for 2 dimensional point) which is a permutation and reflection of the original,
            relative to cell centre. For the second set of vertices, finds the 
            indices of the first index. (which will be a rotation and possible reflection of 
            np.arange(0, len(vertices)))

        Returns None if no valid mapping is found.

        Assumes that vertices and points are numpy arrays
    
    '''
    
    if np.shape(vertices1) != np.shape(vertices2):
        return None
    
    '''
    #Doesn't work with numba
    # Normalizing the vertices (translating to centroid)
    def normalize(vertices):
        centroid = np.mean(vertices, axis=0)
        return vertices - centroid

    if points1 is None:
        v1 = normalize(vertices1)
    else:
        v1 = vertices1 - points1
    
    if points2 is None:
        v2 = normalize(vertices2)
    else:
        v2 = vertices2 - points2
    '''
    
    v1 = vertices1 - points1
    v2 = vertices2 - points2
    
    nv = np.shape(v2)[0]
    shift = 0
    while not allclose(v1[0,:], v2[shift,:]):
        #v2 = np.roll(v2, 1, axis = 0)
        shift += 1
        if shift >= nv:
            return None

    if allclose(v1[1,:], v2[(shift + 1)%nv,:]):
        reverse = False
    elif allclose(v1[1,:], v2[(shift - 1)%nv,:]):
        reverse = True
    else:
        return None
    
    if reverse is False:
        order = (np.arange(0, nv) + shift) % nv
    else:
        order = (nv - np.arange(0, nv) + shift) % nv        

    if allclose(v1, v2[order,:]):
        return order
    else:
        return None    


@jit(nopython=True)
def allclose(a, b, tol = 1e-8):
    return np.all(np.abs(a - b) < tol)

    
def matching_vertex_pairs(vertices1, vertices2, points1, points2):
    ''' 
        Obtains boolean arrays for indices from vertices1 and vertices2 
            of shared vertices
    '''
    v1 = vertices1 - points1
    v2 = vertices2 - points2
    
    v1_ind = (v1[:, None] == v2).all(-1).any(1)
    v2_ind = (v2[:, None] == v1).all(-1).any(1)
    
    return v1_ind, v2_ind
    
    
    
def cyclic_reversed(order):
    ''' 
        Takes ordered array which is a rotation and possible reflection 
            of np.arange(0, len(vertices))
        Returns True if the order is a cyclic reflection
    '''
    
    return not (((order[0] + 1) % len(order)) == order[1])


def next_floor(x):
    '''
        Used to specify the next integer
    '''
    return int(1 + np.floor(x))