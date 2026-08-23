'''
    These provides a way to create the tissue object for pre-defined models 
        (either tissue or tissue0D)
    
    When adding a new model:
        - Add the function
        - Add the name to models0D or models2D
        - Add the number of cell populations to popdict
        
        
    For the governing stem cell division time-scale use tau
    
    For the governing differentiated cell delamination time-scale use mu
    
    For the fraction of asymetric divisions use 1-r

'''

import numpy as np
import sys
sys.path.append("../sim/")
from tissue import *


'''
    0D models:
        e.g. 
        S_0D
        SD_0D    
        STD_0D
        
'''

#List of all models
models0D = ['S_0D', 'SD_0D', 'STD_0D', 'ST2D_0D', 'SPD_0D']
models2D = ['S_mech_del', 'S_mech_div', 'STD_size', 'S_mech_switch',
            'SD_mech_switch', 'ST2D_size', 'SPD_size']

#Dictionary of number of populations per model
popdict = {'S_0D': 1, 'SD_0D': 2, 'STD_0D': 3,
           'S_mech_del': 1, 'S_mech_div': 1,
           'STD_size': 3, 'S_mech_switch': 1,
           'SD_mech_switch': 2,
           'ST2D_0D': 4, 'SPD_0D': 3,
           'ST2D_size': 4, 'SPD_size': 3}


def models(): #list of all models
    return models0D + models2D

def npops(x): #return number of populations
    if x.__name__ in popdict:
        return popdict[x.__name__]
    else:
        print('Model not defined')
        assert(0)
        return 0


def S_0D(ncells, tau, r = 1.0, shed = 0, p = 0.5, useTerm = False, refract = 0.0, 
         sbasrefract = 0, sbas_cv = 1.0, corr = 0.0, cv_div = 1.0):
    '''
        Model with Stem cell dividing at rate tau:
            S-> 2S (prob: r*p)
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p))
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-p)
    '''
      
    if r < 1.0:
        useTerm = True
        
    assert(refract <= 1)
    
    if useTerm == True:
        tiss = tissue0D(poppts = ncells, divtime = (tau - 1)*(1 - refract) + 1, divprob = r*p, assprob = 1 - r, 
                 difprob = 0.0, terprob = r*p, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, popname = ['S'], refract = (tau - 1)*refract,
                 corrSym = corr, cv_div = cv_div)
    else: 
        tiss = tissue0D(poppts = ncells, divtime = (tau - 1)*(1 - refract) + 1, divprob = p, assprob = 0.0, 
                difprob = 1 - p, terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, 
                popname = ['S'], refract = (tau - 1)*refract, cv_div = cv_div)        
    return tiss


def SD_0D(ncells, tau, rho = 0.5, r = 1.0, shed = 0, p = 0.5, mu = None, useTerm = False,
          corrSD = 0.0, corrDD = 0.0, corrSS = 0.0, refractS = 0.0, refractD = 0.0, sbasrefract = 0, sbas_cv = 1.0,
          CVd_S = 1.0, CVd_D = 1.0):
    '''
        Model with stem cell and differentiated cell
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+D (prob: (1-r))
            S->D+D (prob: r*(1-p))
            
        Differentiated cell delaminates at rate mu = tau*(1-rho)/rho
    
        To specify unequal # of initial cells input ncells as a list
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-p)
            
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
        
    '''
    
    if rho == 1.0:
        print('Using S model')
        return S_0D(ncells, tau, r, shed, p, useTerm, refractS, sbasrefract, corrSS)
    
    if r < 1.0:
        useTerm = True   

    if corrDD < 0:
        corrDD = corrSS
    
    if mu is None:
        if useTerm:
            mu = tau*(1-rho)/rho
        else:
            mu = tau*(1-rho)/rho*2         
    else:
        if useTerm:
            rho = 1/(1 + mu/tau)            
        else:
            rho = 1/(1 + 2*mu/tau)   
        
    if np.ndim(ncells) == 0: 
        nS = int(rho*ncells)
        poppts = [ncells - nS, nS]        
    else:   
        poppts = ncells
            
    if useTerm == True:
        tiss = tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                        divprob = [0.0, p*r], assprob = [0.0, 1-r], 
              difprob = [1.0, 0.0], terprob = [0.0, r*(1-p)], sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, popname = ['D', 'S'], 
              corrSym = [corrDD, corrSS], corrAss = corrSD, cv_div = [CVd_D, CVd_S],
              refract = [(mu - 1)*refractD, (tau - 1)*refractS])
    else: 
        tiss = tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                        divprob = [0.0, p], assprob = 0.0, 
              difprob = [1.0, 1-p], terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, popname = ['D', 'S'], 
              corrSym = [corrDD, corrSS], corrAss = corrSD, cv_div = [CVd_D, CVd_S],
              refract = [(mu - 1)*refractD, (tau - 1)*refractS])
              
    return tiss


def STD_0D(ncells, tau, rhoS = 0.25, r = 1.0, corrST = 0.0, corrDD = 0.0, corrSS = 0.0, 
           corrTT = 0.0, shed = 0, p = 0.5, mu = None, refractS = 0.0, refractT = -1, 
           refractD = 0.0, sbasrefract = 0, sbas_cv = 1.0, CVd_S = 1.0, CVd_T = 1.0, CVd_D = 1.0,
           q = 1.0):
    '''
        Model with stem cell, TA cell and differentiated cell
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+T (prob: (1-r))
            S->T+T (prob: r*(1-p))

        TA cell divides at rate tau:
            T->T+D (prob: (1-q))
            T->D+D (prob: q)
            
        Differentiated cell delaminates at rate mu = tau*(1-rhoS)/rhoS
    
        To specify unequal # of initial cells input ncells as a list
        In model #S = #TA
        
        correlations of sisters produced by different different modes are: corrST,corrDD,corrSS,corrTT
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
            we use refractT < 0 to set refractT = refractS            
    '''
    
    if refractT < 0:
        refractT = refractS

    if q < 1.0:
        print('Use STD2D or SPD model instead')
        assert(0)

    if mu is None:
        mu = tau*(1-rhoS*2)/(rhoS*2)
    else:
        rhoS = 1/(1+mu/tau)/2
    
    if np.ndim(ncells) == 0:
        nS = int(rhoS*ncells)
        poppts = [ncells - 2*nS, nS, nS]              
    else:   
        poppts = ncells
    
    assert(rhoS > 0 and rhoS < 1)
        
    return tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractT) + 1, 
              (tau - 1)*(1 - refractS) + 1], divprob = [0.0, 0.0, p*r], 
              assprob = [0.0, 1 - q, 1 - r], difprob = [1.0, 0.0, 0.0], terprob = [0.0, q, r*(1-p)], 
              sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, 
              corrSym = [corrDD, corrTT, corrSS], corrAss = corrST, popname = ['D', 'T', 'S'], cv_div = [CVd_D, CVd_T, CVd_S],
              refract = [(mu - 1)*refractD, (tau - 1)*refractT, (tau - 1)*refractS])




def ST2D_0D(ncells, tau, rhoS = 0.25, r = 1.0, q = 1.0, tauT = None, corrST = 0.0, corrDD = 0.0, corrSS = 0.0, 
           corrTT = 0.0, corrTD = 0.0, shed = 0, p = 0.5, mu = None, refractS = 0.0, refractT = -1, 
           refractD = 0.0, sbasrefract = 0, sbas_cv = 1.0, CVd_S = 1.0, CVd_T = 1.0, CVd_D = 1.0):
    '''
        Model with stem cell, TA cell and differentiated cell, where TA cell 
            can go through one round of asymmetric division before terminal division
        
        Stem cell divides at rate tau:
            S->   2S (prob: r*p)
            S-> S+T1 (prob: (1-r))
            S->T1+T1 (prob: r*(1-p))

        TA1 cell divides at rate tauT:
            T1->T1+D (prob: (1-q))
            T1-> D+D (prob: q)

        TA2 cell terminally divides at rate tauT:
            T2->D+D (prob: 1)
            
        Differentiated cell delaminates at rate mu = tau*(1-rhoS)/rhoS
    
        To specify unequal # of initial cells input ncells as a list
        In model #S = #TA
        
        correlations of sisters produced by different different modes are: corrST,corrDD,corrSS,corrTT
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
            we use refractT < 0 to set refractT = refractS            
    '''
    
    if tauT is None:
        tauT = tau

    if refractT < 0:
        refractT = refractS

    if CVd_T < 0:
        CVd_T = CVd_S

    if corrTT < 0:
        corrTT = corrSS

    if corrTD < 0:
        corrTD = corrST

    if corrDD < 0:
        corrDD = corrSS

    if corrST < 0:
        corrST = corrSS

    if mu is None:
        mu = ((q-2)*rhoS*tauT + tau*(1-rhoS))/(rhoS*(3-q))
        assert(mu > 0)
    else:
        rhoS  =  tau/(tau + ((3 - q)*mu + (2 - q)*tauT))

    rhoT1 = tauT/(tau + ((3 - q)*mu + (2 - q)*tauT))
    rhoT2 = tauT*(1-q)/(tau + ((3 - q)*mu + (2 - q)*tauT))
    rhoD  =   mu*(3-q)/(tau + ((3 - q)*mu + (2 - q)*tauT))
    
    assert(rhoS > 0 and rhoT1 > 0 and rhoT2 >= 0 and rhoD > 0)
    assert(rhoS < 1 and rhoT1 < 1 and rhoT2 < 1 and rhoD < 1)
    assert(np.abs(rhoS + rhoT1 + rhoT2 + rhoD - 1) < 1e-8)

    if np.ndim(ncells) == 0:
        nS  = int( rhoS*ncells)
        nT1 = int(rhoT1*ncells)
        nT2 = int(rhoT2*ncells)
        nD  = int( rhoD*ncells)
        poppts = [nD, nT2, nT1, nS]              
    else:   
        poppts = ncells
          
        
    return tissue0D(poppts = poppts, divprob = [0.0, 0.0, 0.0, p*r], 
              divtime = [(mu - 1)*(1 - refractD) + 1, (tauT - 1)*(1 - refractT) + 1, (tauT - 1)*(1 - refractT) + 1, (tau - 1)*(1 - refractS) + 1], 
              assprob = [0.0, 0.0, 0.0, 1-r], difprob = [1.0, 0.0, 0.0, 0.0], terprob = [0.0, 1.0, q, r*(1-p)], 
              sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, 
              corrSym = [corrDD, corrTT, corrTT, corrSS], corrAss = [corrTD, corrTD, corrST, corrST], popname = ['D', 'T2', 'T1', 'S'],
              cv_div = [CVd_D, CVd_T, CVd_T, CVd_S], refract = [(mu - 1)*refractD, (tauT - 1)*refractT, (tauT - 1)*refractT, (tau - 1)*refractS],
              daughterTypes = [4, 0, 0, 2], cusdivprobs = [0.0, 0.0,[1-q],0], cusdivDaughters = [0,0,[[0,1]],0])




def SPD_0D(ncells, tau, tauP = None, rhoS = 0.25, r = 1.0, delta = 0.0, q = 1, corrSP = 0.0, corrDD = 0.0, 
           corrSS = 0.0, corrPP = 0.0, corrPD = 0.0, shed = 0, p = 0.5, mu = None,
           sbasrefract = 0, sbas_cv = 1.0, CVd_S = 1.0, CVd_P = 1.0, CVd_D = 1.0,
           refractS = 0.0, refractP = 0.0, refractD = 0.0):
    '''
        Model with stem cell, progenitor and differentiated cell
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+P (prob: (1-r))
            S->P+P (prob: r*(1-p))
      
        Progenitor cell divides at rate tauP:
            P-> 2P (prob: q*delta)
            P->P+D (prob: (1-q))
            P->D+D (prob: q*(1-delta))            
            
        Differentiated cell delaminates at rate mu
    
        To specify unequal # of initial cells input ncells as a list
        
        correlations of sisters produced by different different modes are: corrST,corrDD,corrSS,corrTT
            
    '''

    if tauP is None:
        tauP = tau

    if refractP < 0:
        refractP = refractS

    if CVd_P < 0:
        CVd_P = CVd_S 

    if CVd_P < 0:
        CVd_P = CVd_S

    if corrSP < 0:
        corrSP = corrSS

    if corrPP < 0:
        corrPP = corrSS

    if corrPD < 0:
        corrPD = corrSP

    if corrDD < 0:
        corrDD = corrSS               

    if mu is None:
        mu   = (q*(1 - rhoS)*(-1 + 2*delta)*tau + rhoS*tauP)/(rhoS*(-1 + q*(-1 + 2*delta)))
    else:
        rhoS = (q*(1 - 2*delta)*tau)/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)

    rhoP = tauP/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)
    rhoD = ((1 + q*(1 - 2*delta))*mu)/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)

    assert(rhoS > 0 and rhoP > 0 and rhoD > 0)
    assert(rhoS < 1 and rhoP < 1 and rhoD < 1)

    if np.ndim(ncells) == 0:
        nS = int(rhoS*ncells)
        nT = int(rhoP*ncells)
        nD = int(rhoD*ncells)
        poppts = [nD, nT, nS]              
    else:   
        poppts = ncells    

    return tissue0D(poppts = poppts, divtime = [(mu - 1)*(1 - refractD) + 1, (tauP - 1)*(1 - refractP) + 1, 
              (tau - 1)*(1 - refractS) + 1], divprob = [0.0, q*delta, p*r], 
              assprob = [0.0, 1-q, 1-r], difprob = [1.0, 0.0, 0.0], terprob = [0.0, q*(1-delta), r*(1-p)], 
              sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), 
              sbas_cv = sbas_cv, corrSym = [corrDD, corrPP, corrSS], 
              corrAss = [0.0, corrPD, corrSP], popname = ['D', 'P', 'S'], cv_div = [CVd_D, CVd_P, CVd_S],
              refract = [(mu - 1)*refractD, (tauP - 1)*refractP, (tau - 1)*refractS])


'''
    Spatial models:
        e.g.
        S_mech_del
        S_mech_div 
        STD_size

        
    
'''

def S_mech_del(ncells, L, tau, r = 1.0, shed = 0, tadiv = 10, Alow = 0.75, useTerm = False, sbasrefract = 0, sbas_cv = 1.0):
    ''' 
        Basal stem cell with division drives delamination
        
        Area rule on delamination
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p)
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p))        

        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-p)        
    '''
      
    if r < 1.0:
        useTerm = True
    
    if useTerm == True:
        tiss = tissue(poppts = ncells, L = L, divtime = tau/(1 - r/2), divprob = r/(1-r/2)/2, assprob = (1 - r)/(1-r/2), Alow = Alow, Ahigh = np.inf,
                 difprob = 0.0, terprob = r*p, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, rule = 'size', areadivtime = tadiv, popname = ['S'])
    else: 
        tiss = tissue(poppts = ncells, L = L, divtime = tau, divprob = 1.0, assprob = 0.0, Alow = Alow, Ahigh = np.inf,
                difprob = 0.0, terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, areadivtime = tadiv, popname = ['S'])        
    return tiss



def S_mech_div(ncells, L, tau, shed = 0, tadiv = 10, Ahigh = 1.25, sbasrefract = 0, sbas_cv = 1):
    ''' 
        Basal stem cell with delamination drives division
        
        Area rule on delamination
            S-> 2S (A > Ac)
            S->u (rate 1/tau)
    
    '''
      
    return tissue(poppts = ncells, L = L, divtime = tau, divprob = 1.0, assprob = 0.0, Alow = 0.0, Ahigh = Ahigh,
                 difprob = 1.0, terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, rule = 'size', areadivtime = tadiv, popname = ['S'])


def S_mech_switch(ncells, L, tau = None, r = 1.0, shed = 0, tadiv = 10, Ahigh = np.inf, hilln = 2.0, 
                  rho_c = 1.0,  switch_profile = 'binary', useTerm = False, rlambda = None, sbasrefract = 0, 
                  cv_div = 1.0, sbas_cv = 1, Ly = None):
    ''' 
        Basal stem cell with division at fixed rate 1/tau1 (for A < Ahigh) and 1/tau2 for (A > Ahigh)
            can use other switch types (e.g. binary, relu)
        
        Area controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+u (prob: (1-r))
            S->u+u (prob: r*(1-p(A*rhoc)))        
            
        If useTerm = False then we use a model without terminal divisions:
            S->u (prob: 1-(1-p(A*rhoc))   
            
            
        If rlambda is specified, then we use r*lambda is fixed and then r is varied, i.e. the value of tau is not used
    '''
      
    if r < 1.0:
        useTerm = True
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)               
    
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
    else:
        ruleS = 'size-size'
        assert(0) # Not implemented in this paper
    
    if useTerm == True:
        tiss = tissue(poppts = ncells, L = L, Ly = Ly, divtime = tau, divprob = r/2, assprob = 1-r, Alow = 0.0, Ahigh = Ahigh,
                 difprob = 0.0, terprob = r/2, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, rule = ruleS, areadivtime = tadiv, popname = ['S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, cv_div = cv_div)
    else: 
        tiss = tissue(poppts = ncells, L = L, Ly = Ly, divtime = tau, divprob = 0.5, assprob = 0.0, Alow = 0.0, Ahigh = Ahigh,
                 difprob = 0.5, terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, rule = ruleS, areadivtime = tadiv, popname = ['S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, cv_div = cv_div)
    return tiss


def SD_mech_switch(ncells, L, tau, r = 1.0, rho = 0.5, shed = 0, tadiv = 10, Ahigh = np.inf, hilln = 2.0, 
                  rho_c = 1.0, mu = None, switch_profile = 'binary', useTerm = False,
                  mechDelam = False, Alow = 0.0, A0 = 1.0, D_A0 = 0.0,
                  corrSD = 0.0, corrDD = 0.0, corrSS = 0.0, refractS = 0.0, refractD = 0.0, 
                  rlambda = None, sbasrefract = 0, sbas_cv = 1.0, CVd_S = 1.0, CVd_D = 1.0, **kwargs):
    ''' 
        Basal stem cell with division at fixed rate 1/tau1 (for A < Ahigh) and 1/tau2 for (A > Ahigh)
            can use other switch types (e.g. binary, relu)
        
        Area controls the fate balance, p(A*rhoc) = hill(A*rhoc, hilln)
        
        Stem cell dividing at rate tau:
            S-> 2S (prob: r*p(A*rhoc))
            S->S+D (prob: (1-r))
            S->D+D (prob: r*(1-p(A*rhoc)))        
            
        Differentiated cell delaminates at rate mu = tau*(1-rho)/rho            
            
        If useTerm = False then we use a model without terminal divisions:
            S->D (prob: 1-(1-p(A*rhoc)) 
            
        Use mechDelam = True for a mechanical rule on delamination (D->u)
            need to specify Alow (defaults to 0.05)
            can also specify A0 (defaults to 1.0)
        
    '''

    
    if rho == 1.0:
        print('Using S model')
        return S_mech_switch(ncells, L, tau, r, shed, tadiv, Ahigh, hilln, rho_c, switch_profile, useTerm, rlambda = rlambda, sbasrefract = sbasrefract)


    #Set the rule on Stem-cells
    Alows = Alow 
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
    else:
        ruleS = 'size-size'
        assert(0) # Not implemented in this paper

    if corrDD < 0:
        corrDD = corrSS    

    if CVd_D < 0:
        CVd_D = CVd_S
            
      
    #Set the rule on differentiated cells
    if mechDelam == True:
        if Alow == 0.0:
            Alow = 0.05
        Alows = [Alow, 0.0]
        ruleD = 'size'
        mu = 0.0 
        print('Using mechanical delamination rule, mu ~ 100')
        sim_params = {'A0': [D_A0, A0]}
    else:
        ruleD = 'stochastic'
        sim_params = None 
      
    if r < 1.0:
        useTerm = True   
        
    if rlambda is not None:
        tau = r/rlambda
        print('Using fixed r*lambda = ', rlambda, ', tau = ', tau, ', varying r = ', r)         
    
    if mu is None:
        if useTerm:
            mu = tau*(1-rho)/rho
        else:
            mu = tau*(1-rho)/rho*2
    else:
        if mu == 0:
            mu1 = 100
        else:
            mu1 = mu
            
        if useTerm:
            rho = 1/(1+mu1/tau)            
        else:
            rho = 1/(1+2*mu1/tau)
        
    if np.ndim(ncells) == 0: 
        nS = int(rho*ncells)
        poppts = [ncells - nS, nS]        
    else:   
        poppts = ncells
    
    if useTerm == True:
        tiss = tissue(poppts = poppts, L = L, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1], 
                      divprob = [0,r/2], assprob = [0,1-r], Ahigh = [np.inf, Ahigh],
                 difprob = [1.0, 0.0], terprob = [0.0,r/2], sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), 
                 sbas_cv = sbas_cv, rule = [ruleD, ruleS], areadivtime = tadiv, popname = ['D','S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, Alow = Alows,
                 sim_params = sim_params, corrSym = [corrDD, corrSS], corrAss = corrSD,
                 refract = [(mu - 1)*refractD, (tau - 1)*refractS],
                 cv_div = [CVd_D, CVd_S], **kwargs)
    else: 
        tiss = tissue(poppts = poppts, L = L, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractS) + 1],
                      divprob = [0.0, 0.5], assprob = 0.0, Ahigh = [np.inf, Ahigh],
                 difprob = [1.0, 0.5], terprob = 0.0, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), 
                 sbas_cv = sbas_cv, rule = [ruleD, ruleS], areadivtime = tadiv, popname = ['D','S'],
                 rho_c = rho_c, hilln = hilln, profile = switch_profile, Alow = Alows, 
                 sim_params = sim_params, corrSym = [corrDD, corrSS], corrAss = corrSD,
                 refract = [(mu - 1)*refractD, (tau - 1)*refractS], cv_div = [CVd_D, CVd_S], **kwargs)
        
    return tiss



def STD_size(ncells, L, tau, rhoS = 0.5, r = 1.0, hilln = 2.0,  corrST = 0.0, corrDD = 0.0, corrSS = 0.0, corrTT = -1,
                    shed = 0, mu = None, rho_c = 1.0, refractS = 0.0, refractT = -1, refractD = 0.0,
                    mechDelam = False, Alow = 0.0, A0 = 1.0, tadiv = 10.0, sbasrefract = 0, sbas_cv = 1.0,
                    CVd_S = 1.0, CVd_T = 1.0, CVd_D = 1.0, switch_profile = 'binary', Ahigh = np.inf, D_A0 = 0.0):
    '''
        Model with stem cell, TA cell and differentiated cell
        
        rhoS is the density of stem cells
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+T (prob: (1-r))
            S->T+T (prob: r*(1-p))
            
        Differentiated cell delaminates at rate mu = tau*(1-rho)/rho
    
        To specify unequal # of initial cells input ncells as a list
        In model #S = #TA
        
        correlations of sisters produced by different different modes are: corrST,corrDD,corrSS,corrTT
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
            we use refractT < 0 to set refractT = refractS               
            
            
        Use mechDelam = True for a mechanical rule on delamination (D->u)
            need to specify Alow (defaults to 0.05)
            can also specify A0 (defaults to 1.0)
                        
    '''

    if refractT < 0:
        refractT = refractS

    if corrTT < 0:
        corrTT = corrSS
        
    #Set the rule on Stem-cells
    Alows = Alow 
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
        ruleT = 'stochastic'
    else:
        ruleS = 'size-size'    
        ruleT = 'size'    
        assert(0) # Not implemented in this paper        
        
    #Set the rule on differentiated cells
    if mechDelam == True:
        if Alow == 0.0:
            Alow = 0.05
        Alows = [Alow, 0.0, 0.0]
        ruleD = 'size'
        mu = 0.0 
        print('Using mechanical delamination rule, mu ~ 100')
        sim_params = {'A0': [D_A0, A0, A0]}
    else:
        ruleD = 'stochastic'
        sim_params = None         
        

    if mu is None:
        mu = tau*(1-rhoS*2)/(rhoS*2)
    else:
        if mu == 0:
            mu1 = 100
        else:
            mu1 = mu        
        rhoS = 1/(1+mu1/tau)/2
        
    assert(mu >= 0)
    assert(rhoS > 0 and rhoS < 1)
    
    if np.ndim(ncells) == 0:
        nS = int(rhoS*ncells)
        poppts = [ncells - 2*nS, nS, nS]        
    else:   
        poppts = ncells
        
    if corrTT is None:
        corrTT = corrSS
        
    return tissue(poppts = poppts,  L = L, divtime = [(mu - 1)*(1 - refractD) + 1, (tau - 1)*(1 - refractT) + 1, 
              (tau - 1)*(1 - refractS) + 1], divprob = [0.0, 0.0, r/2], assprob = [0.0, 0.0, 1 - r], 
                  difprob = [1.0, 0.0, 0.0], Alow = Alows, Ahigh = [np.inf, Ahigh, Ahigh],
            terprob = [0.0, 1.0, r/2], conserve = None, rule = [ruleD, ruleT, ruleS],
            hilln = hilln, rho_c = rho_c, corrAss = corrST, sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, 
            corrSym = [corrDD, corrTT, corrSS], sim_params = sim_params, areadivtime = tadiv,
            popname = ['D', 'T', 'S'], refract = [(mu - 1)*refractD, (tau - 1)*refractT, (tau - 1)*refractS],
            cv_div = [CVd_D, CVd_T, CVd_S], profile = switch_profile)
    


def SPD_size(ncells, L, tau, tauP = None, rhoS = 0.5, r = 1.0, delta = 0.0, q = 1.0,
                hilln = 2.0, corrSP = 0.0, corrDD = 0.0, shed = 0, mu = None, rho_c = 1.0,
                corrSS = 0.0, corrPP = 0.0, corrPD = 0.0, refractS = 0.0, refractP = -1, refractD = 0.0,
                mechDelam = False, Alow = 0.0, A0 = 1.0, tadiv = 10.0, sbasrefract = 0, sbas_cv = 1.0,
                CVd_S = 1.0, CVd_P = 1.0, CVd_D = 1.0, switch_profile = 'binary', Ahigh = np.inf, D_A0 = 0.0):
    '''
        Model with stem cell, progenitor and differentiated cell
        
        Stem cell divides at rate tau:
            S-> 2S (prob: r*p)
            S->S+P (prob: (1-r))
            S->P+P (prob: r*(1-p))
      
        Progenitor cell divides at rate tauP:
            P-> 2P (prob: q*delta)
            P->P+D (prob: (1-q))
            P->D+D (prob: q*(1-delta))            
            
        Differentiated cell delaminates at rate mu
    
        To specify unequal # of initial cells input ncells as a list

        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
            we use refractT < 0 to set refractT = refractS               
            
            
        Use mechDelam = True for a mechanical rule on delamination (D->u)
            need to specify Alow (defaults to 0.05)
            can also specify A0 (defaults to 1.0)
                        
    '''
    
    if tauP is None:
        tauP = tau

    if refractP < 0:
        refractP = refractS

    if CVd_P < 0:
        CVd_P = CVd_S 

    if CVd_P < 0:
        CVd_P = CVd_S

    if corrPP < 0:
        corrPP = corrSS

    if corrPD < 0:
        corrPD = corrSP

    if corrDD < 0:
        corrDD = corrSS               

    if mu is None:
        mu   = (q*(1 - rhoS)*(-1 + 2*delta)*tau + rhoS*tauP)/(rhoS*(-1 + q*(-1 + 2*delta)))
    else:
        rhoS = (q*(1 - 2*delta)*tau)/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)

    rhoP = tauP/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)
    rhoD = ((1 + q*(1 - 2*delta))*mu)/(q*(1 - 2*delta)*tau + (1 + q*(1 - 2*delta))*mu + tauP)

    assert(rhoS > 0 and rhoP > 0 and rhoD > 0)
    assert(rhoS < 1 and rhoP < 1 and rhoD < 1)

    if np.ndim(ncells) == 0:
        nS = int(rhoS*ncells)
        nT = int(rhoP*ncells)
        nD = int(rhoD*ncells)
        poppts = [nD, nT, nS]              
    else:   
        poppts = ncells   

    #Set the rule on Stem-cells
    Alows = Alow 
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
        ruleP = 'stochastic'
    else:
        ruleS = 'size-size'    
        ruleP = 'size'  
        assert(0) # Not implemented in this paper         

    #Set the rule on differentiated cells
    if mechDelam == True:
        if Alow == 0.0:
            Alow = 0.05
        Alows = [Alow, 0.0, 0.0]
        ruleD = 'size'
        mu = 0.0 
        print('Using mechanical delamination rule, mu ~ 100')
        sim_params = {'A0': [D_A0, A0, A0]}
    else:
        ruleD = 'stochastic'
        sim_params = None         
    
    p = 1/2

    return tissue(poppts = poppts,  L = L, 
              divtime = [(mu - 1)*(1 - refractD) + 1, (tauP - 1)*(1 - refractP) + 1, 
              (tau - 1)*(1 - refractS) + 1], divprob = [0.0, q*delta, p*r], 
              assprob = [0.0, 1-q, 1-r], difprob = [1.0, 0.0, 0.0], terprob = [0.0, q*(1-delta), r*(1-p)], 
              sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), 
              sbas_cv = sbas_cv, corrSym = [corrDD, corrPP, corrSS], 
              corrAss = [0.0, corrPD, corrSP], popname = ['D', 'P', 'S'], cv_div = [CVd_D, CVd_P, CVd_S],
              refract = [(mu - 1)*refractD, (tauP - 1)*refractP, (tau - 1)*refractS],       
              Alow = Alows, Ahigh = [np.inf, Ahigh, Ahigh],
              conserve = None, rule = [ruleD, ruleP, ruleS],
              hilln = hilln, rho_c = rho_c, sim_params = sim_params, areadivtime = tadiv,
              profile = switch_profile)
    





def ST2D_size(ncells, L, tau, rhoS = 0.5, r = 1.0, q = 1.0, hilln = 2.0,  corrST = 0.0, corrDD = 0.0, corrSS = 0.0, corrTT = -1,
                    corrTD = 0.0, shed = 0, mu = None, rho_c = 1.0, refractS = 0.0, refractT = -1, refractD = 0.0, tauT = None,
                    mechDelam = False, Alow = 0.0, A0 = 1.0, tadiv = 10.0, sbasrefract = 0, sbas_cv = 1.0,
                    CVd_S = 1.0, CVd_T = 1.0, CVd_D = 1.0, switch_profile = 'binary', Ahigh = np.inf, D_A0 = 0.0):
    '''
        Model with stem cell, TA cell and differentiated cell
        
        rhoS is the density of stem cells
        
        Stem cell divides at rate tau:
            S->   2S (prob: r*p)
            S-> S+T1 (prob: (1-r))
            S->T1+T1 (prob: r*(1-p))

        TA1 cell divides at rate tauT:
            T1->T1+D (prob: (1-q))
            T1-> D+D (prob: q)

        TA2 cell terminally divides at rate tauT:
            T2->D+D (prob: 1)
            
        Differentiated cell delaminates at rate mu = tau*(1-rhoS)/rhoS
    
        To specify unequal # of initial cells input ncells as a list
        In model #S = #TA
        
        correlations of sisters produced by different different modes are: corrST,corrDD,corrSS,corrTT
            
        We can encode refractory periods for each cell type as a fraction of decision tims
            refractory period of S = (tau - 1)*refractS
            refractory period of D =  (mu - 1)*refractD
            we use refractT < 0 to set refractT = refractS               
            
            
        Use mechDelam = True for a mechanical rule on delamination (D->u)
            need to specify Alow (defaults to 0.05)
            can also specify A0 (defaults to 1.0)
                        
    '''

    if tauT is None:
        tauT = tau

    if refractT < 0:
        refractT = refractS

    if CVd_T < 0:
        CVd_T = CVd_S

    if corrTT < 0:
        corrTT = corrSS

    if corrTD < 0:
        corrTD = corrST

    if corrDD < 0:
        corrDD = corrSS

    if mu is None:
        mu = ((q-2)*rhoS*tauT + tau*(1-rhoS))/(rhoS*(3-q))
        assert(mu > 0)
    else:
        rhoS  =  tau/(tau + ((3 - q)*mu + (2 - q)*tauT))

    rhoT1 = tauT/(tau + ((3 - q)*mu + (2 - q)*tauT))
    rhoT2 = tauT*(1-q)/(tau + ((3 - q)*mu + (2 - q)*tauT))
    rhoD  =   mu*(3-q)/(tau + ((3 - q)*mu + (2 - q)*tauT))
    
    assert(rhoS > 0 and rhoT1 > 0 and rhoT2 >= 0 and rhoD > 0)
    assert(rhoS < 1 and rhoT1 < 1 and rhoT2 < 1 and rhoD < 1)
    assert(np.abs(rhoS + rhoT1 + rhoT2 + rhoD - 1) < 1e-8)

    if np.ndim(ncells) == 0:
        nS  = int( rhoS*ncells)
        nT1 = int(rhoT1*ncells)
        nT2 = int(rhoT2*ncells)
        nD  = int( rhoD*ncells)
        poppts = [nD, nT2, nT1, nS]              
    else:   
        poppts = ncells

    
        
    #Set the rule on Stem-cells
    Alows = Alow 
    if Ahigh == np.inf:
        ruleS = 'size-fixrate'
        ruleT = 'stochastic'
    else:
        ruleS = 'size-size'    
        ruleT = 'size'  
        assert(0) # Not implemented in this paper          
        
    #Set the rule on differentiated cells
    if mechDelam == True:
        if Alow == 0.0:
            Alow = 0.05
        Alows = [Alow, 0.0, 0.0]
        ruleD = 'size'
        mu = 0.0 
        print('Using mechanical delamination rule, mu ~ 100')
        sim_params = {'A0': [D_A0, A0, A0]}
    else:
        ruleD = 'stochastic'
        sim_params = None         
        
    p = 1/2

    return tissue(poppts = poppts,  L = L, 
            divprob = [0.0, 0.0, 0.0, p*r], 
            divtime = [(mu - 1)*(1 - refractD) + 1, (tauT - 1)*(1 - refractT) + 1, (tauT - 1)*(1 - refractT) + 1, (tau - 1)*(1 - refractS) + 1], 
            assprob = [0.0, 0.0, 0.0, 1-r], difprob = [1.0, 0.0, 0.0, 0.0], terprob = [0.0, 1.0, q, r*(1-p)], 
            sbasshed = (shed - 1)*(1 - sbasrefract) + 1, sbasrefract = sbasrefract*(shed - 1), sbas_cv = sbas_cv, 
            corrSym = [corrDD, corrTT, corrTT, corrSS], corrAss = [corrTD, corrTD, corrST, corrST], popname = ['D', 'T2', 'T1', 'S'],
            cv_div = [CVd_D, CVd_T, CVd_T, CVd_S], refract = [(mu - 1)*refractD, (tauT - 1)*refractT, (tauT - 1)*refractT, (tau - 1)*refractS],
            daughterTypes = [4, 0, 0, 2], cusdivprobs = [0.0, 0.0,[1-q],0], cusdivDaughters = [0,0,[[0,1]],0],
            hilln = hilln, rho_c = rho_c, rule = [ruleD, ruleT, ruleT, ruleS],
            sim_params = sim_params, areadivtime = tadiv, profile = switch_profile,
            Alow = Alows, Ahigh = [np.inf, Ahigh, Ahigh, Ahigh])


