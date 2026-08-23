'''
    Plotting suite for tissue

'''

#Use: import tissplot as tplt

from tissue import *
import numpy as np
import matplotlib.pyplot as plt
import analysis
import random

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage






def plotBCD(results, dens0 = 1, plotpops = True, plottot = True, tscale = 1, tiss = None, tshift = 0):
    #Reference density: dens0, if dens0 = 0 then we use time-average of total # of cells
    #scale = 1/dt if you want time, or 1/rate

    colors  = ['0.75', 'g', 'b', 'r', 'm', 'k']

    npops = np.shape(results['ncells'])[1]

    if dens0 == 0:
        dens0 = np.mean(np.sum(results['ncells'],1))

    ntimes = np.shape(results['ncells'])[0]
    timevec = np.arange(ntimes)/tscale + tshift

    labels = []
    if tiss is None:
        plabels = list(np.arange(npops))
    else:
        plabels = tiss.popnames()

    #plt.figure()
    if plotpops is True:
        for p in range(npops):
            plt.plot(timevec, results['ncells'][:,p]/dens0, color = colors[p])
        labels += plabels
    if plottot is True:
        plt.plot(timevec, np.sum(results['ncells']/dens0, 1), 'k-')
        labels += ['total']
    
    plt.legend(labels)
    plt.ylabel('Basal cell density')
    plt.xlabel('Time')

    ax = plt.gca()
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')


def plot_av_cs():
    return

def plot_cs_dist(cdf = False):
    return


def clonal_snapshot(tiss, results, times = [-1], track = 4, sample_mode = -10, cloneResults = None, 
                    skip = None, titles = None, markpops = True, use_legend = True, col = None,
                    fillclones = True, fillpops = False, popcols = None, print_labelled = True,
                    colorbyval = None, vmin = None, vmax = None, cmap = 'viridis', cmap_title = '',
                    scale = 1, xlim = None, ylim = None, markersize = 5, cbarticks = None,
                    title_font_size = 16, cbar_font_size = 12, cbar_tick_labels = None,
                    lw = 5):
    #Plots a snapshot of tissue across different times with the same clones tracked
    '''
        Pass in:
               tiss: tissue object
            results: results struct
            clone_results: clone_results object if you have it, otherwise will re-calculate
            
            ntimes: list of times, if negative numbers determines from reverse snapshots
        
            track:
                if type is  int: is number of clones to track and we randomly sample based on sample mode
                if type is list: track specific clones

            sample_mode
                 n: For n >= 0, samples from all cells from a specific cell population
              -n+1: For n < npops, samples from all surviving cells from a that population 
                    at latest time-point plotted
                -10: sample from all clones
                -20: sample from all surviving clones at latest time-point plotted
                -30: sample from extinct clones

        colorbyval: must be a list of length ntimes with each entry a list of length #cells at that time
            to color cells by
        vmin, vmax: color scale limits
    '''
    
    if col is None:
        col = ['r', 'g', 'y', 'b', 'm', 'c', 'k', 'tab:orange', 'gray']

    if popcols is None:
        popcols = ['w', 'b', 'r', 'g', 'm', 'k']       
    
    ntimes = len(results['ncells'])
    L = tiss.L
    
    if np.min(times) < 0: #repair times, so can specifiy points at end
        tarr = np.array(times, dtype = 'int')
        tarr[tarr < 0] = np.arange(0, ntimes, skip)[tarr[tarr < 0]]
        times = list(tarr)

    if skip is None: #calculate skip if specified
        skip = analysis.calculateSkip(results)
        if np.isnan(skip):
            print('Skip not defined')
            assert(0)
            
    if cloneResults is None:
        cloneResults = analysis.clonestats(results, tiss, tvals = None, skip = skip)

    if titles is None:
        titles = [str(t) for t in times]
        
    positions = results['cellpos']
    ancs = results['anc']
    cellid = results['cellid']
    ncells = np.sum(results['ncells'],1)
    
    survivors = analysis.survivors(cloneResults, time = times[-1])
    diers     = analysis.diers(cloneResults,     time = times[-1])
    labeled   = cellid[0]
    
    #get population ids for initial cells
    ncells0 = results['ncells'][0,:]
    splits = list(np.cumsum(ncells0))
    splits = np.array([0] + splits)
    
    cellpopids = [cellid[0][splits[i]:splits[i+1]] for i in range(tiss.npops)] 
    
    ncl   =  len(survivors)    
    
    
    if type(track) is not list:
        if track > 0:
            if sample_mode == -10:
                samplefrom = labeled            
            elif sample_mode == -20:
                samplefrom = survivors
            elif sample_mode == -30:
                samplefrom = diers
            elif np.abs(sample_mode) < tiss.npops + 1:
                if sample_mode < 0:
                    sample_mode = np.abs(sample_mode + 1)
                    samplefrom = np.array(list(set(survivors).intersection(set(cellpopids[sample_mode]))))
                else:
                    samplefrom = cellpopids[sample_mode]
            else:
                track = 0
                samplefrom = np.array([])
            if track > 0:
                track = list(samplefrom[np.array(random.sample(range(len(samplefrom)), track))])
            else:
                track = []
        else:
            track = []
            
    
    #Plotting part    
    plt.figure(figsize = (5.5*len(times)*scale,6*scale), dpi = 300)
    
    for tt in range(len(times)):
        t = times[tt]
        ts = int(t/skip)
        plt.subplot(1,len(times),tt + 1)
        t1 = tissue(positions[ts] % L, L = L)
        
        splits = list(np.cumsum(results['ncells'][t,:]))
        splits = np.array([0] + splits)
        
        ax = plt.gca()
        voronoi_plot_2d(t1, ax, show_vertices = False, show_points = False)
        

        if fillpops:
            for p in range(tiss.npops):
                for c in range(splits[p],splits[p+1]):
                    region = t1.regions[t1.point_region[c]]
                    polygon = [t1.vertices[i] for i in region]
                    plt.fill(*zip(*polygon), popcols[p], alpha = 0.7) 

        if colorbyval is not None:
            if vmin is None:
                vmin = np.min(colorbyval[ts])
            if vmax is None:
                vmax = np.max(colorbyval[ts])
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)
            for c in range(len(positions[ts])):
                region = t1.regions[t1.point_region[c]]
                polygon = [t1.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color = cmap_obj(norm(colorbyval[ts][c])), alpha = 0.7) 
            #Colorbar is just added to the last plot, below

        
        #Mark clones
        if len(track) > 0:
            clones = cloneResults['allclones'][t]
            for j in range(len(track)):
                if print_labelled:
                    print('Clone', track[j], col[j])
                for c in clones[track[j]]:
                    region = t1.regions[t1.point_region[c]]
                    polygon = [t1.vertices[i] for i in region]
                    if fillclones:
                        plt.fill(*zip(*polygon), col[j%len(col)], alpha = 0.55)
                    else:
                        ax.fill(*zip(*polygon), facecolor=(1,0,0,0), edgecolor = col[j%len(col)], linewidth = lw*scale)
        
        #Mark cell populations
        if markpops:
            markers = ['o',  '^', 'o', '*', 'h', 'x']
            if tiss.npops == 2:
                colors  = ['w', 'k']
            elif tiss.npops == 1:
                colors = ['k']
            else:
                colors  = popcols     
            
            if tiss.npops > 1:
                plt.plot(positions[ts][splits[0]:splits[1],0] % L, positions[ts][splits[0]:splits[1],1] % L, markers[0],
                    label = tiss.cellpops[0].popname, color = colors[0], markersize = 0.000001)
                p0 = 1
            else:
                p0 = 0

            for p in range(p0,tiss.npops):
                plt.plot(positions[ts][splits[p]:splits[p+1],0] % L, positions[ts][splits[p]:splits[p+1],1] % L, markers[0],
                    label = tiss.cellpops[p].popname, color = colors[p], markersize = markersize)

            if tiss.npops > 1 and use_legend:
                plt.legend()


                
        #plt.title('t = ' + str(titles[tt]))
        plt.title(titles[tt], fontsize = title_font_size)
            
        plt.axis('square')   
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim([0, L])
        if ylim is not None:
            plt.ylim(ylim)
        else:
            plt.ylim([0, L])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.01)
    
    
    if colorbyval is not None:
        #cbar = plt.colorbar(cax = (vmin,vmax), ax = ax, fraction=0.046, pad=0.04)
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig = plt.gcf()
        fig.subplots_adjust(right=1 - 0.2/len(times))
        cbar_ax = fig.add_axes([(len(times) - 1)/(len(times)), 0.1, 1/(len(times)), 0.8])
        '''
            Note adding axis will put cells in the wrong spot if you edit
        '''

        cbar_ax.axis('off')
        #cbar_ax.set_yticks([])
        #cbar_ax.set_xticks([])        
        cbar = fig.colorbar(sm, ax = cbar_ax)
        cbar.ax.tick_params(labelsize=None) 
        cbar.set_label(cmap_title, fontsize = cbar_font_size)

        if cbarticks is not None:
            cbar.set_ticks(cbarticks, labels = cbar_tick_labels)

        cbar.ax.tick_params(labelsize=cbar_font_size)


def mosaic_snapshot(tiss, results, times = [-1], tracks = None, sample_mode = -10, cloneResults = None, 
                    skip = None, titles = None, markpops = True, use_legend = True, col = None):
    #Plots a snapshot of tissue across different times with the same clones tracked
    '''
        Pass in:
               tiss: tissue object
            results: results struct
            clone_results: clone_results object if you have it, otherwise will re-calculate
            
            ntimes: list of times, if negative numbers determines from reverse snapshots
        
            track:
                if type is  int: is number of clones to track and we randomly sample based on sample mode
                if type is list: track specific clones

            sample_mode
                 n: For n >= 0, samples from all cells from a specific cell population
              -n+1: For n < npops, samples from all surviving cells from a that population 
                    at latest time-point plotted
                -10: sample from all clones
                -20: sample from all surviving clones at latest time-point plotted
                -30: sample from extinct clones
    '''
    
    if col is None:
        col = ['w', 'b', 'r', 'g', 'y', 'm', 'c', 'k']
    
    ntimes = len(results['ncells'])
    L = tiss.L
    
    if np.min(times) < 0: #repair times, so can specifiy points at end
        tarr = np.array(times, dtype = 'int')
        tarr[tarr < 0] = np.arange(0, ntimes, skip)[tarr[tarr < 0]]
        times = list(tarr)

    if skip is None: #calculate skip if specified
        skip = analysis.calculateSkip(results)
        if np.isnan(skip):
            print('Skip not defined')
            assert(0)
            
    if cloneResults is None:
        cloneResults = analysis.clonestats(results, tiss, tvals = None, skip = skip)

    if titles is None:
        titles = [str(t) for t in times]
        
    positions = results['cellpos']
    ancs = results['anc']
    cellid = results['cellid']
    ncells = np.sum(results['ncells'],1)
    
    survivors = analysis.survivors(cloneResults, time = times[-1])
    diers     = analysis.diers(cloneResults,     time = times[-1])
    labeled   = cellid[0]
    
    #get population ids for initial cells
    ncells0 = results['ncells'][0,:]
    splits = list(np.cumsum(ncells0))
    splits = np.array([0] + splits)
    
    cellpopids = [cellid[0][splits[i]:splits[i+1]] for i in range(tiss.npops)] 
    
    ncl   =  len(survivors)    


    track = list(tracks.keys())
    
    #Plotting part    
    plt.figure(figsize = (6*len(times),6), dpi = 300)
    for tt in range(len(times)):
        t = times[tt]
        ts = int(t/skip)
        plt.subplot(1,len(times),tt + 1)
        t1 = tissue(positions[ts] % L, L = L)
        
        splits = list(np.cumsum(results['ncells'][t,:]))
        splits = np.array([0] + splits)
        
        ax = plt.gca()
        voronoi_plot_2d(t1, ax, show_vertices = False, show_points = False)
        
        #Mark clones
        if len(track) > 0:
            clones = cloneResults['allclones'][t]
            for j in range(len(track)):
                print('Clone', track[j], col[tracks[track[j]]])
                for c in clones[track[j]]:
                    region = t1.regions[t1.point_region[c]]
                    polygon = [t1.vertices[i] for i in region]
                    plt.fill(*zip(*polygon),  col[tracks[track[j]]], alpha = 0.5)
        
        
        #Mark cell populations
        if markpops:
            markers = ['o',  '^', 'o', '*', 'h', 'x']
            if tiss.npops <= 2:
                colors  = ['w', 'k']
            else:
                colors  = ['w', 'b', 'r', 'g', 'm', 'k']            
            plt.plot(positions[ts][splits[0]:splits[1],0] % L, positions[ts][splits[0]:splits[1],1] % L, markers[0],
                label = tiss.cellpops[0].popname, color = colors[0], markersize = 0.000001)

            for p in range(1,tiss.npops):
                plt.plot(positions[ts][splits[p]:splits[p+1],0] % L, positions[ts][splits[p]:splits[p+1],1] % L, markers[0],
                    label = tiss.cellpops[p].popname, color = colors[p])

            if tiss.npops > 1 and use_legend:
                plt.legend()
                
        #plt.title('t = ' + str(titles[tt]))
        plt.title(titles[tt], fontsize = 16)
            
        plt.axis('square')    
        plt.xlim([0, L])
        plt.ylim([0, L])
        plt.xticks([])
        plt.yticks([])
    

    
    
def clone_movies(positions, ancs = None, cellid = None, ncells = None, nsteps = 100, savename = 'sim.mp4', 
                 gifname = None, track = [0,1,2], fps = 10, L = 15, highlight = None, 
                 skip = 1, diebytime = None, delambytime = None, tscale = 1, tshift = 0, 
                 bridges = None, saveskip = 1, col = None, markpops = True,
                    fillclones = True, fillpops = False, popcols = None, 
                    use_legend = True, production = False):    


    if col is None:
        col = ['r', 'g', 'y', 'b', 'm', 'c', 'k']

    if popcols is None:
        popcols = ['w', 'b', 'r', 'g', 'm', 'k'] 
    
    # to plot animation use: animation.ipython_display(fps = fps, loop = True, autoplay = True)
    print(len(track))
    # Store the original track parameter
    original_track = track.copy() if isinstance(track, list) else track
    col = [plt.cm.hsv(i) for i in range(0,200,np.round(200/len(original_track)).astype('int'))]
    random.shuffle(col)
    
    # matplot subplot
    if production:
        #fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # 4:3, 1600x1200 px
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # 640x480 px
    else:
        fig, ax = plt.subplots()

    # method to get frames
    def make_frame(t):
        # Make sure we use the original track parameter
        nonlocal original_track

        # clear
        ax.clear()
        t = int(t*skip*fps)
        
        # Create Voronoi diagram first
        t1 = Voronoi(addPBC(positions[t] % L, L))
        voronoi_plot_2d(t1, ax, show_vertices = False, show_points = False)
        
        if ancs is not None:
            cloneidx = [ancs[i] for i in cellid[t][:np.sum(ncells[int(t*saveskip)])]]
            clones = {i: [] for i in cellid[0][:np.sum(ncells[0])]}
        
            for i in range(len(cloneidx)):
                clones[cloneidx[i]].append(i)
            
            # Process tracking for cells
            for j in range(len(original_track)):
                if original_track[j] in clones:  # Make sure the tracked cell exists
                    for c in clones[original_track[j]]:
                        region = t1.regions[t1.point_region[c]]
                        polygon = [t1.vertices[i] for i in region]
                        ax.fill(*zip(*polygon), c = col[j])
        
        # t1 already created above
        
        if highlight is not None:
            plt.plot(highlight[:,0], highlight[:,1], 'k.', markersize = 15)

        ncells0 = ncells[int(t*saveskip),:]
        splits = list(np.cumsum(ncells0))
        splits = np.array([0] + splits)
    
        '''
        if ncells is not None:
            #highlight population 2 (SC's)
            plt.plot(positions[t][ncells[int(t*saveskip),0]:,0] % L, positions[t][ncells[int(t*saveskip),0]:,1] % L, 'k.', markersize = 15)            
        '''
        tiss = t1
        npops = ncells.shape[1] #4
        ts = int(t)
        popnames = ['D','TA','TA','SC']
        #Mark cell populations
        if markpops:
            markers = ['o',  '^', 'o', '*', 'h', 'x']
            if npops <= 2:
                colors  = ['w', 'k']
            else:
                colors  = popcols     
            plt.plot(positions[ts][splits[0]:splits[1],0] % L, positions[ts][splits[0]:splits[1],1] % L, markers[0],
                label = popnames[0], color = colors[0], markersize = 0.000001)

            for p in range(1,npops):
                plt.plot(positions[ts][splits[p]:splits[p+1],0] % L, positions[ts][splits[p]:splits[p+1],1] % L, markers[0],
                    label = popnames[p], color = colors[p], markersize = 7)

            if npops > 1 and use_legend:
                plt.legend()


        #Mark dying cells
        if diebytime is not None:
            for c in diebytime[t*saveskip]:
                #print(c, t, diebytime[t])
                where = np.argwhere(cellid[t] == c)
                if len(where) > 0: 
                    cid = np.argwhere(cellid[t] == c)[0][0]
                    #print(cid, positions[t][cid])
                    plt.plot(positions[t][cid,0] % L, positions[t][cid,1] % L, 'rx', markersize = 10)

        #Mark delaminating cells
        if delambytime is not None:
            for c in delambytime[t*saveskip]:
                where = np.argwhere(cellid[t] == c)
                if len(where) > 0: 
                    cid = where[0][0]
                    plt.plot(positions[t][cid,0] % L, positions[t][cid,1] % L, 'g^', markersize = 10)
                
        if bridges is not None:
            for c in cellid[t][:ncells[int(t*saveskip),0]]:
                if c in bridges:
                    csis = bridges[c] 
                    cid = np.argwhere(cellid[t] == c)[0][0]
                    if csis in cellid[t]:
                        csisid = np.argwhere(cellid[t] == csis)[0][0]
                        pair = np.array([cid,csisid])
                        if np.max(np.abs(positions[t][cid,:] % L - positions[t][csisid,:] % L)) < L/2:
                            plt.plot(positions[t][pair,0] % L, positions[t][pair,1] % L, 'g--')
            
        
        plt.title('t = '+ "{:.1f}".format(t*saveskip/tscale + tshift) + ' days')
        ax.axis('square')
        ax.set_xlim([0,L])
        ax.set_ylim([0,L])
        plt.xticks([])
        plt.yticks([])
        
        # returning numpy image
        return mplfig_to_npimage(fig)

    

    # creating animation
    animation = VideoClip(make_frame, duration = nsteps/fps) # ntsteps/20/10
    
    if savename is not None:
        """
        if production:
            animation.write_videofile(
                savename,
                fps=fps,
                codec='libx264',
                ffmpeg_params=[
                    '-crf', '0',
                    '-preset', 'veryslow',
                    '-pix_fmt', 'yuv444p',
                ],
            )
        """
        animation.write_videofile(
            savename,
            fps=fps,
            codec='libx264',
            ffmpeg_params=[
                '-crf', '15',           # visually high quality, not lossless
                '-preset', 'slow',
                '-pix_fmt', 'yuv420p',
            ],
        )
    else:
        animation.write_videofile(savename, fps=fps)
        
    if gifname is not None:
        animation.write_gif(gifname, fps=fps)        

    return animation


def addPBC(points, L, rinc = None): #rinc is inclusion radius for PBCs
    if rinc == None:
        rinc = L
    vx = L*np.array((1,0), dtype = np.float64)
    vy = L*np.array((0,1), dtype = np.float64)
    left  = points[:,0] < rinc
    right = points[:,0] > L - rinc
    up    = points[:,1] > L - rinc
    down  = points[:,1] < rinc
    points = np.vstack([points, 
                        points[left ].copy() + vx, 
                        points[right].copy() - vx, 
                        points[up   ].copy() - vy, 
                        points[down ].copy() + vy, 
                        points[left  & down].copy() + vx + vy, 
                        points[right & up  ].copy() - vx - vy, 
                        points[left  & up  ].copy() + vx - vy, 
                        points[right & down].copy() - vx + vy])
    return points
