'''


    Order parameters for simulations are functions on dataset objects (ultimately based 
        on quantities or distributions) which yield either a scalar or vector.
        
    An OP can be specified as a tuple (op, type, args) 
        with the following types:
            - 'all': stores the entire time series
            - 'mean': stores the mean of the time series
            - 'std': stores the standard deviation of the time series
            - 'sum': stores the sum of the time series
            - 'max': stores the maximum of the time series
            - 'min': stores the minimum of the time series`
            - 'av_rate': stores the number of events per time unit (in simulation time)
            - 'av_rate_per_cell': stores the number of events per time unit per cell
            - 'slope': stores the slope of the time series using linear regression
                    then start_time is an argument
            
    Then additional arguments depend on the specific order parameter at play
    
    To express units of time set tscalevals (e.g. tscale = -1, will normalise according to tau)
    
    These numbers are combined across seeds
    
    These order parameters can be stored in search_space
    
    
    The only things we need to modify to add a new order parameter are:
        - the op_categories dictionary
        - the op_from_ds function
'''

import numpy as np
from scipy import stats
import datasets as ds

#Valid-types
valid_types = ['all', 'mean', 'std', 'sum', 'max', 'min', 'av_rate', 'av_rate_per_cell', 
               'slope']

#Dictionary setting the dataset category for each order parameter
op_categories = {'T1s': ['T1s'],
                 'MSD': ['MSD'],
                 'shape': ['shape'],
                 'ncells': ['density'],
                 'poppts': ['density'], 
                 'reff': ['fate_reff'],
                 'area': ['area']}


def op_from_ds(dset, op):
    '''
        Calculate order parameters from dataset objects
        
        Assumes that the order parameter is valid
        
        Gets the time-vector for simulations
    '''
    
    #assert(dset.seeded    is False)
    assert(dset.collapsed is True)
    
    if len(op) == 1:
        op = (op,'all')
    
    if len(op) > 2:
        args = op[2]
        
    else:
        args = {}
        
    #Specify the right data and times from dataset
    if op[0] == 'T1s':
        data = dset.quantities['T1s']
    elif op[0] == 'MSD':
        data = dset.quantities['MSD']
    elif op[0] == 'shape':
        data = dset.quantities['shape']
    elif op[0] == 'ncells':
        data = dset.quantities['density']
    elif op[0] == 'poppts':
        data = dset.pop_quantities['density']
        if 'axis' not in args:
            args['axis'] = 1
    elif op[0] == 'reff':
        data = dset.timeless['fate_reff']
    elif op[0] == 'area':
        data = dset.quantities['area']
    else:
        raise ValueError('Order parameter not recognised')
        
    #Get times associated with data
    times = dset.tvec_by_quantity(op_categories[op[0]][0])
    
    #From this data we calculate the order parameter
    if op[1] == 'all':
        return data
    elif op[1] == 'mean':
        return np.mean(data, **args)
    elif op[1] == 'std':
        return np.std(data, **args)
    elif op[1] == 'sum':
        return np.sum(data, **args)
    elif op[1] == 'max':
        return np.max(data, **args)
    elif op[1] == 'min':
        return np.min(data, **args)
    elif op[1] == 'av_rate':
        dt = (times[-1] - times[0])/(len(times) - 1)
        return np.sum(data, **args)/(dt*len(data))
    elif op[1] == 'av_rate_per_cell':
        dt = (times[-1] - times[0])/(len(times) - 1)
        ncells = dset.quantities['density']
        if ds.skipped[op_categories[op[0]][0]] == True:
            ncells = ncells[::dset.skip]  
        return np.sum(data/ncells, **args)/(dt*len(data))
    elif op[1] == 'slope':
        if len(op) > 2:
            start_time = op[2]
        else:
            start_time = 0
        slope, intercept, r_value, p_value, std_err = stats.linregress(times[start_time:], data[start_time:])
        return slope




def op_names(ops):
    return [op_name(op) for op in ops]




def ops_from_ds(dset, ops):
    ''' 
        Given a list of order parameters, calculates all order parmaeters
    
    '''    
    
    if ops is None:
        return {}
    
    allops = {}
    assert(type(ops) is list)
    
    for op in ops:
        assert(check_valid_op(op))
        allops[op_name(op)] = op_from_ds(dset, op)
        
    return allops
        
    
    
    
def op_name(op):
    ''' 
        Returns a string for order parameter name
    '''
    return op[0] + '_' + op[1]



def check_valid_op(op): 
    '''
        Checks the order parameter specified is valid
    '''
    
    if type(op) is not tuple:
        print('Order parameter is not a tuple')
        return False
    
    if len(op) < 2:
        print('Order parameter wrong format')
        return False
    
    if op[0] in op_categories:
        if op[1] in valid_types:
            return True
        else:
            print('Order parameter type is invalid')
            return False
    else:
        print('Order parameter is invalid')
        return False


def getToCalculate(ops):
    ''' 
        Gets the categories to calculate given order parameters
    '''
    
    if ops is None:
        return []

    toCalculate = []
    
    for op in ops:
        assert check_valid_op(op) , op
        toCalculate.extend(op_categories[op[0]])
    
    toCalculate = list(set([o for op in ops for o in op_categories[op[0]]]))
    types = [op[1] for op in ops]
    
    if 'av_rate_per_cell' in types:
        toCalculate.append('density')
        
    return list(set(toCalculate))