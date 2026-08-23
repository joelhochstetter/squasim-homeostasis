#####################################################
#Plotting functionality for datasets
#####################################################


import matplotlib.pyplot as plt
import numpy as np
import datasets as ds


def plot_quantity(q, sim_data = None, exp_data = None, tscale = None, color = 'k', ylabel = None,
                  plot_seeded = False, prelabel = '', bootSEs = False, errmult = 1.0, sim_shift = 0.0,
                  sim_mult = 1.0, sim_slice = None, nolegend = False, sfmt = '-', xfmt = 'o'):
    '''
        Plots a quantity q, comparing a simulation to experimental data
        
        
        Specify:
            color
            ylabel
            inc_seeded = True => include seeded datapoints
            
        sim_slice: enter [start, end] to slice the simulation data [default = None]
        
    '''
    
    if sim_slice is None:
        sim_slice = slice(None)
    else:
        sim_slice = slice(sim_slice[0], sim_slice[1])
    
    #Set-up defaults
    if ylabel is None:
        ylabel = q
        
    if tscale is None:
        tscale = sim_data.tscalevals[0]
    
    #Initialise simulated data
    if sim_data is not None:
        assert(sim_data.collapsed is True)
        
    #Plotting simulation data
    if sim_data is not None:
        tvec = sim_data.tvec_by_quantity(q, tscale)
        if sim_data.seeded is False:
            if bootSEs is True and q in sim_data.dists:
                sim_data.bootstrap_SEs(q, exp_data.nsamples[q])
            plt.plot(sim_shift + tvec[sim_slice], sim_data.quantities[q][sim_slice]*sim_mult, sfmt, color = color, label = prelabel)
            if q in sim_data.se_quants:
                plt.fill_between(sim_shift + tvec[sim_slice], (sim_data.quantities[q][sim_slice] - sim_data.se_quants[q][sim_slice]*errmult)*sim_mult, 
                    (sim_data.quantities[q][sim_slice] + sim_data.se_quants[q][sim_slice]*errmult)*sim_mult, color = color, alpha = 0.3, label = '_nolegend_')
            
        else: #seeded data
            sd1 = sim_data.calculate_combined_quantities(full = True)
            if bootSEs is True and q in sim_data.dists:
                sd1.bootstrap_SEs(q, exp_data.nsamples[q])            
            plt.plot(sim_shift + tvec[sim_slice], sd1.quantities[q][sim_slice]*sim_mult, sfmt, color = color, label = prelabel)
            if q in sd1.se_quants:
                plt.fill_between(sim_shift + tvec[sim_slice], (sd1.quantities[q][sim_slice] - sd1.se_quants[q][sim_slice]*errmult)*sim_mult, 
                    (sd1.quantities[q][sim_slice] + sd1.se_quants[q][sim_slice]*errmult)*sim_mult, color = color, alpha = 0.3, label = '_nolegend_')            
            if plot_seeded:
                plt.plot(sim_shift + tvec[:,None], sim_data.quantities[q].transpose()*sim_mult, sfmt, color = color, alpha = 0.3)
    
    #Plotting experimental data
    if exp_data is not None:
        if q in exp_data.se_quants:
            plt.errorbar(exp_data.timeq(q), exp_data.quantities[q], yerr = exp_data.se_quants[q]*errmult, fmt = xfmt, label = '_nolegend_', color = color)
        else:
            plt.plot(exp_data.timeq(q), exp_data.quantities[q], xfmt, label = '_nolegend_', color = color)
    
    plt.xticks()
    plt.yticks()
    plt.xlabel('Time (days)')
    plt.ylabel(ylabel)
    if nolegend is False:
        plt.legend(loc = 'best', frameon = False)
    



def plot_dist(d, sim_data = None, exp_data = None, tscale = None, color = 'k', xlabel = None,
                  plot_seeded = False, prelabel = '', subtimes = None, subplot_axs = None,
                  im_width = 4, maxbin = None, cdf = False, bootSEs = False, overhang_in_last = True,
                  noticks = False, errmult = 1.0, ncols = 3, nolegend = False, ymax = None, xmax = None,
                  ylabel = None, yticks = None, yscale = 1.5, nogaps = False,
                  num_xticks = None, add_vlines = None, x0 = 0):
    '''
        Plots a distribution d, comparing a simulation to experimental data
        
        By default plots pdf, usless cdf = True then plots cdf
        
        If subplot_axs is None, creates a new sub-plot
        
        Specify:
            color
            ylabel
            inc_seeded = True => include seeded datapoints
            
            subtimes: indicates the indices of the experimental time-point to use      
            
            overhang_in_last: is whether last bin in pdf shown contains all higher clone sizes  
    '''    
    
    #Set-up defaults
    if xlabel is None:
        xlabel = d

    
    #Initialise simulated data
    if sim_data is not None:
        assert(sim_data.collapsed is True)
        #extract at time-points
        sdt = sim_data.data_time_crop()
        
    #Set-up timepoints
    if exp_data is not None:
        exp_data = exp_data.copy() #so manipulations don't ruin distribution
        times  = exp_data.timeq(d)
        if d in exp_data.max_pdf_bin:
            maxpbin = exp_data.max_pdf_bin[d]
        else:
            maxpbin = 200
    elif sim_data is not None:
        times = sdt.timeq(d)
        maxpbin = sim_data.max_pdf_bin[d]
        if d in sim_data.max_pdf_bin:
            maxpbin = sim_data.max_pdf_bin[d]
        else:
            maxpbin = 200        
    else:
        print('No data provided')
        assert(0)
        
    if maxbin is None:
        maxbin = maxpbin
        
    if subtimes is not None:
        times = times[subtimes]
    else:
        subtimes = np.arange(len(times))
        
    ntimes = len(times)        
    nrows = int(np.ceil(ntimes/ncols))
        
    
    #Create sub-plots
    if subplot_axs is None:
        fig, axs = plt.subplots(nrows, ncols, figsize=(im_width*ncols, yscale*im_width*nrows), dpi=400)    
    else:
        axs = subplot_axs
        
    #Set-up dist
    if sim_data is not None:
        if sim_data.seeded is False:
            sim_dists = sdt.reshape_pdf(d, max_bin = maxbin + (1 - int(overhang_in_last)))
            if bootSEs is True:
                se_dists = sdt.bootstrap_SEs(d, exp_data.nsamples[d], sedists = True)*errmult          
        else:
            sd1 = sdt.calculate_combined_quantities(full = True)
            sim_dists = sd1.reshape_pdf(d, max_bin = maxbin + (1 - int(overhang_in_last)))
            if bootSEs is True:
                se_dists = sd1.bootstrap_SEs(d, exp_data.nsamples[d], sedists = True)*errmult    
                    
            if plot_seeded:
                seeded_dists = sdt.reshape_pdf(d, max_bin = maxbin + (1 - int(overhang_in_last)))
                if cdf is True:
                    seeded_dists = ds.getCdf(seeded_dists)
                    
        if cdf is True:
            sim_dists = ds.getCdf(sim_dists)
    
    if exp_data is not None:
        exp_dists = exp_data.reshape_pdf(d, max_bin = maxbin + (1 - int(overhang_in_last)))
        
        if bootSEs is True:
            exp_se_dists = exp_data.bootstrap_SEs(d, exp_data.nsamples[d], sedists = True)*errmult
        
        if cdf is True:
            exp_dists = ds.getCdf(exp_dists)
    
    for t in range(ntimes):
        xx = int(np.floor((t + x0)/ncols))
        yy = (t + x0) % ncols 
        tt = subtimes[t]
        
        if ntimes > ncols:
            ax = axs[xx,yy]
        elif ncols > 1:
            ax = axs[yy]
        else:
            ax = axs
            
        #Plotting simulation data
        if sim_data is not None:
            if sim_data.seeded is False:
                ax.plot(np.arange(0, maxbin + 1), sim_dists[tt, 0:(maxbin+1)], '-', label = prelabel, color = color, alpha = 0.8)
                if bootSEs is True:
                    ax.fill_between(np.arange(0, maxbin + 1), sim_dists[tt, 0:(maxbin+1)] - se_dists[tt, 0:(maxbin+1)], 
                        sim_dists[tt, 0:(maxbin+1)] + se_dists[tt, 0:(maxbin+1)], color = color, alpha = 0.3)
            else: #seeded data             
                ax.plot(np.arange(0, maxbin + 1), sim_dists[tt, 0:(maxbin+1)], '-', label = prelabel, color = color, alpha = 0.8)
                if bootSEs is True: 
                    ax.fill_between(np.arange(0, maxbin + 1), sim_dists[tt, 0:(maxbin+1)] - se_dists[tt, 0:(maxbin+1)], 
                        sim_dists[tt, 0:(maxbin+1)] + se_dists[tt, 0:(maxbin+1)], color = color, alpha = 0.3)
                if plot_seeded:                     
                    ax.plot(np.arange(0, maxbin + 1)[None, :], seeded_dists[:, tt, 0:(maxbin+1)], '-', label = prelabel, color = color, alpha = 0.8)
        
        #Plotting experimental data
        if exp_data is not None:   
            if bootSEs is True:
                ax.errorbar(np.arange(0, maxbin + 1), exp_dists[tt, 0:(maxbin+1)], exp_se_dists[tt, 0:(maxbin+1)], fmt = 'o', label = '_nolegend_', color = color)
            else:
                ax.plot(np.arange(0, maxbin + 1), exp_dists[tt, 0:(maxbin+1)], 'o', label = '_nolegend_', color = color)
        
        ax.set_title('Day ' + str(int(times[t]))) 

        if add_vlines is not None:
            if np.ndim(ymax) > 0:
                for vl in add_vlines:
                    ax.addvlines(vl, 0, ymax[t], color = 'k', linestyle = '-', alpha = 0.1)
            else:
                for vl in add_vlines:
                    ax.vlines(vl, 0, ymax, color = 'k', linestyle = '-', alpha = 0.1)

        if ymax is not None:
            if np.ndim(ymax) > 0:
                ax.set_ylim([0,ymax[t]])
            else:
                ax.set_ylim([0,ymax])

        if xmax is not None:
            if np.ndim(xmax) > 0:
                ax.set_xlim([0,xmax[t]])
                if num_xticks is not None:
                    xticks = np.linspace(0, xmax[t], num_xticks).astype(int)
                    if yy < ncols - 1:
                        xticks = xticks[:-1]
                    else:
                        xticks = xticks[:]
                    ax.set_xticks(xticks)                
            else:
                ax.set_xlim([0,xmax])
                if num_xticks is not None:
                    xticks = np.linspace(0, xmax, num_xticks).astype(int)
                    if yy < ncols - 1:
                        xticks = xticks[:-1]
                    else:
                        xticks = xticks[:]                        
                    ax.set_xticks(xticks)

        
        if nolegend is False:       
            ax.legend(loc = 'upper right', frameon = False)
        
        if noticks:
            ax.set_xticks([])
            ax.set_yticks([])

        if yticks is not None:
            if np.ndim(yticks) > 0 and np.ndim(yticks[0]) > 0:
                ax.set_yticks(yticks[t])
            else:
                if yy == 0:
                    ax.set_yticks(yticks)
                else:
                    ax.set_yticks([])
        
        if yy == 0:
            if ylabel is not None:
                ax.set_ylabel(ylabel)     
            else:
                if cdf:
                    ax.set_ylabel('P(' + xlabel + ' < X)')
                else:
                    ax.set_ylabel('P(' + xlabel + ')')
        
        if yy == 0:
            ax.set_xlabel(xlabel)

        

    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()

    if nogaps:
        plt.subplots_adjust(wspace=0, hspace=0)

    
    return axs