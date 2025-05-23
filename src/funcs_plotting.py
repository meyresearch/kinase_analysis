# Functions for plotting MSM results

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import networkx as nx
from deeptime.plots import plot_implied_timescales, plot_energy2d, plot_contour2d_from_xyz
from deeptime.util import energy2d
import numpy as np

dfg_spatial_colors = np.array(['#595959',        # Grey noise
                               '#540D6E',        # Purple DFG-in
                               '#EE4266',        # Red DFG-inter
                               '#FFD23F'])       # Yellow DFG-out

# dfg_spatial_colors = np.array(['#595959',       # Grey noise
#                                '#53F4A2',       # Green DFG-in
#                                '#F54801',       # Red DFG-inter
#                                '#BA08F4'])      # Purple DFG-out

dfg_dihed_colors = np.array(['#595959',         # Grey noise
                             '#A9E67B',
                             '#00E617',
                             '#1F6B40',
                             '#1FBCE6',
                             '#7088E6',
                             '#150AE6',         
                             '#F48263',          # Inter
                             '#F36BF5'])         # Out

def plot_ev(ev, c_centers, traj_all, traj_weights, title, savedir, dim_1=0, dim_2=1, dim_3=2, \
            ct_cmap='nipy_spectral', ct_a=0.6, ev_cmap='coolwarm', ev_a=0.8, ev_s=20, ev_marker='.', \
            ex=True, ex_s=100):

    vmin, vmax = min(ev), max(ev)
    divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    min_id, max_id = ev.argmin(), ev.argmax()

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_axes([0, 0, 0.5, 0.5])
    ax2 = fig.add_axes([0.52, 0, 0.5, 0.5])
    fe_cax = fig.add_axes([ax1.get_position().x0-0.08, ax1.get_position().y0, 0.03, ax1.get_position().height])
    ev1_cax = fig.add_axes([ax1.get_position().x0, ax1.get_position().y1+0.02, ax1.get_position().width, 0.03])
    ev2_cax = fig.add_axes([ax2.get_position().x0, ax2.get_position().y1+0.02, ax2.get_position().width, 0.03])

    ax1, contour1, cbar1 = plot_energy2d(energy2d(traj_all[:,dim_1], traj_all[:,dim_2], weights=traj_weights), ax=ax1, cbar_ax=fe_cax, contourf_kws=dict(cmap=ct_cmap, alpha=ct_a))
    ax2, contour1, cbar2 = plot_energy2d(energy2d(traj_all[:,dim_3], traj_all[:,dim_2], weights=traj_weights), ax=ax2, cbar=False, contourf_kws=dict(cmap=ct_cmap, alpha=ct_a))

    ax1_eg2 = ax1.scatter(c_centers[:,dim_1], c_centers[:,dim_2], s=ev_s, c=ev, marker=ev_marker, alpha=ev_a, cmap=ev_cmap, norm=divnorm)
    fig.colorbar(ax1_eg2, cax=ev1_cax, format='%.1f', orientation='horizontal')

    ax2_eg2 = ax2.scatter(c_centers[:,dim_3], c_centers[:,dim_2], s=ev_s, c=ev, marker=ev_marker, alpha=ev_a, cmap=ev_cmap, norm=divnorm)
    fig.colorbar(ax2_eg2, cax=ev2_cax, format='%.1f', orientation='horizontal')
    
    if ex:
        ax1.scatter(c_centers[min_id,dim_1], c_centers[min_id,dim_2], s=ex_s, c='k', marker='X', label='min')
        ax1.scatter(c_centers[max_id,dim_1], c_centers[max_id,dim_2], s=ex_s, c='k', marker='v', label='max')
        ax2.scatter(c_centers[min_id,dim_3], c_centers[min_id,dim_2], s=ex_s, c='k', marker='X', label='min')
        ax2.scatter(c_centers[max_id,dim_3], c_centers[max_id,dim_2], s=ex_s, c='k', marker='v', label='max')
        ax1.legend()
        ax2.legend()

    ev1_cax.tick_params(axis='x', bottom=False, top=True, labeltop=True, labelbottom=False)
    ev2_cax.tick_params(axis='x', bottom=False, top=True, labeltop=True, labelbottom=False)
    ax2.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)
    fe_cax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)

    ax1.set_xlabel(f'tIC {dim_1+1}', fontsize=14)
    ax2.set_xlabel(f'tIC {dim_3+1}', fontsize=14)
    ax2.set_ylabel(f'tIC {dim_2+1}', fontsize=14)
    ax2.yaxis.set_label_position("right")
    ev1_cax.set_xlabel(title, fontsize=14)
    ev2_cax.set_xlabel(title, fontsize=14)
    ev1_cax.xaxis.set_label_position('top')
    ev2_cax.xaxis.set_label_position('top')
    fe_cax.set_ylabel('Free energy (kT)', fontsize=14)
    fe_cax.yaxis.set_label_position("left")

    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_fe(traj_all, traj_weights, savedir=None, fes_cmap='nipy_spectral', 
            dim_1 = 0, dim_2 = 1, \
            c_centers=None, d_centers=None, 
            c_centers_s=10, c_centers_marker='.', c_centers_a=0.8, c_centers_c='black',
            d_centers_s=10, d_centers_marker='X', d_centers_a=0.8, d_centers_c='black', d_edgecolor='white', d_linewidth=1,
            state_assignment=None, n_states= None, state_population=None, pcca_cmap='gist_rainbow', edgecolor='black', linewidth=1,
            legend_marker_sizes=100,
            title=''):
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax, contour, cbar = plot_energy2d(energy2d(traj_all[:, dim_1], traj_all[:, dim_2], weights=traj_weights), ax=ax, contourf_kws=dict(cmap=fes_cmap))
    
    if state_assignment is not None:
        colours = [plt.cm.get_cmap(pcca_cmap)(i/(n_states-1)) for i in range(n_states)]
        if state_population is not None:
            for i in range(n_states):
                ax.scatter(c_centers[state_assignment == i, dim_1], c_centers[state_assignment == i, dim_2], 
                        s=c_centers_s, c=colours[i], marker=c_centers_marker, alpha=c_centers_a, 
                        edgecolor=edgecolor, linewidth=linewidth,
                        label=f'macrostate {i+1} ({state_population[i]*100:.1f}%)')
            legend = plt.legend(markerscale=1, loc='best', fontsize=10)
        else:
            for i in range(n_states):
                ax.scatter(c_centers[state_assignment == i, dim_1], c_centers[state_assignment == i, dim_2], 
                        s=c_centers_s, c=colours[i], marker=c_centers_marker, alpha=c_centers_a, 
                        edgecolor=edgecolor, linewidth=linewidth,
                        label=f'macrostate {i+1}')

    if (state_assignment is  None) and (c_centers is not None): 
        ax.scatter(c_centers[:,dim_1], c_centers[:,dim_2], s=c_centers_s, c=c_centers_c, marker=c_centers_marker, alpha=c_centers_a)
    
    if d_centers is not None:
        ax.scatter(d_centers[:,dim_1], d_centers[:,dim_2], s=d_centers_s, c=d_centers_c, marker=d_centers_marker, alpha=d_centers_a, 
                   edgecolor=d_edgecolor, linewidth=d_linewidth, label='disconnected states')
        legend = plt.legend(markerscale=1, loc='best', fontsize=10)

    ax.set_xlabel(f'tIC {dim_1+1}', fontsize=14)
    ax.set_ylabel(f'tIC {dim_2+1}', fontsize=14)
    cbar.ax.set_ylabel('Free energy (kT)', fontsize=14)
    if title is not None: ax.set_title(title, fontsize=16)

    try:
        if type(legend_marker_sizes) is int: legend_marker_sizes = [legend_marker_sizes] * len(legend.legend_handles)
        for i, handle in enumerate(legend.legend_handles):
            handle.set_sizes([legend_marker_sizes[i]])
    except:
        pass

    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_pcca_graph(traj_all, traj_weights,  
                    c_centers, matrix, pcca_assignment, stat_dist, 
                    fes_cmap='nipy_spectral', dim_1 = 0, dim_2 = 1, 
                    c_centers_s=50, c_centers_marker='.', c_centers_a=0.5, 
                    pcca_cmap='gist_rainbow', c_edgecolor='black', linewidth=1, 
                    g_alpha=0.8, connectionstyle='Angle3', 
                    savedir=None):
    
    n_states = len(np.unique(pcca_assignment))
    colours = [cm.get_cmap(pcca_cmap)(i/(n_states-1)) for i in range(n_states)]
    
    # Compute the centeroids of pcca macrostates 
    # Using the microstate centers weighted by the stationary probability
    macrostate_centroid = []
    for i in range(n_states):
        macrostate_cluster_centers = c_centers[pcca_assignment == i, :]
        macrostate_cluster_weights = stat_dist[pcca_assignment == i]
        weighted_centroid = np.average(macrostate_cluster_centers, axis=0, weights=macrostate_cluster_weights)
        macrostate_centroid.append(weighted_centroid)
    pos = {i:(macrostate_centroid[i][dim_1], macrostate_centroid[i][dim_2]) for i in range(n_states)}

    # Node sizes are scaled by the stationary probability of the macrostate
    cg_stationary_dist = [sum(stat_dist[pcca_assignment == i]) for i in range(n_states)]
    node_sizes = cg_stationary_dist/min(cg_stationary_dist)*100
    node_ln_widths = np.cbrt(node_sizes/min(node_sizes))
    node_ft_sizes = np.cbrt(node_sizes/min(node_sizes))*8


    fig, ax = plt.subplots(figsize=(11, 10))
    # Plot the FES as a background
    ax, contour, cbar = plot_energy2d(energy2d(traj_all[:, dim_1], traj_all[:, dim_2], weights=traj_weights), ax=ax, contourf_kws=dict(cmap=fes_cmap))
    # Plot the microstate centers coloured by pcca assignment
    for i in range(n_states):
        ax.scatter(c_centers[pcca_assignment == i, dim_1], c_centers[pcca_assignment == i, dim_2], 
                    s=c_centers_s, c=colours[i], marker=c_centers_marker, alpha=c_centers_a, 
                    edgecolor=c_edgecolor, linewidth=linewidth)

    ax.set_xlabel(f'tIC {dim_1+1}', fontsize=14)
    ax.set_ylabel(f'tIC {dim_2+1}', fontsize=14)
    cbar.ax.set_ylabel('Free energy (kT)', fontsize=14)

    G = nx.DiGraph()
    # Add nodes and edges to the graph
    for i in range(n_states):
        G.add_node(i, label=f'{i+1}')
        for j in range(n_states):
            if (i != j) and (matrix[i, j] < 3000000):
                G.add_edge(i, j, weight=np.log2(max(matrix.flatten()) / matrix[i, j]))
    edge_widths = [edge[2]['weight'] for edge in G.edges(data=True)]

    nx.draw(G, pos, 
            node_size=node_sizes[[node[0] for node in G.nodes(data=True)]],
            node_color=[colours[node[0]] for node in list(G.nodes(data=True))], 
            edgecolors='black', linewidths=node_ln_widths[[node[0] for node in G.nodes(data=True)]], 
            alpha=g_alpha, 
            edge_color = [colours[edge[0]] for edge in G.edges(data=True)], 
            width=edge_widths, arrows=True, connectionstyle=connectionstyle,
            ax=ax)
    # Add labels to the nodes
    node_labels = nx.get_node_attributes(G, 'label')
    for node, (x, y) in pos.items():
        label = node_labels[node]
        ax.text(x, y, label, fontsize=node_ft_sizes[int(node)], ha='center', va='center')

    plt.tight_layout(pad=2.0)
    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_ts(timescales, n_ts, dt, savedir):
    '''
    Plot implied timescales of processes from a Maximum likihood MSM or Bayesian MSM samples. 
    
    Parameters
    ----------
    timescales : np.array
        Array of timescales from the MSM samples. If timescales has shape (n_markov_states), plot the single measurement. If timescales has shape (n_markov_states, n_samples), plot the mean and std over n_samples. 
    n_ts : int
        Number of timescales to plot.
    dt : float
        The time interval of the feature trajectories in nanoseconds. Should be consistent with 
        dt_out in the study object. 
    savedir : str
        Path to save the plot. If None, the plot will be displayed but not saved.
    '''
    
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(n_ts)+2
    if timescales.ndim == 1:
        y = timescales[:n_ts]*dt/1000
        if len(timescales) < n_ts: UserWarning("Not enough timescales to plot.")
        ax.plot(x, y, marker='o', markersize=10)
    elif timescales.ndim == 2:
        y = timescales.mean(axis=0)[:n_ts]*dt/1000
        yerr = timescales.std(axis=0)[:n_ts]*dt/1000
        if timescales.shape[1] < n_ts: UserWarning("Not enough timescales to plot.")
        ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=5, capsize=6, capthick=2, ls='-', ecolor='black')
    else:
        raise ValueError("timescales must have shape (n_markov_states) or (n_markov_states, n_samples)")
    
    ax.fill_between([0, n_ts+2], y1=dt/1000, y2=0, color='grey', alpha=0.8)

    ax.set_yscale('log')
    ax.grid(visible=True, axis='y')

    ax.set_xlim([1, n_ts+2])
    ax.set_ylim([min(y)/3, max(y)*3])

    ax.set_xticks(np.arange(2, n_ts+2, 1))
    ax.set_ylabel(r"Timescale ($\mathrm{\mu s}$)", fontsize=14)
    ax.set_xlabel(r"Timescale Index", fontsize=14)
    ax.tick_params(bottom=True, top=False, left=True, right=False)

    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_pcca(state_assignment, c_centers, savedir, dim_1=0, dim_2=1, \
              c_centers_s=6, cmap='gist_rainbow'):
    
    n_states = len(np.unique(state_assignment))
    cmap = mpl.colormaps[f'{cmap}'].resampled(n_states)
    norm = colors.BoundaryNorm(list(range(0, n_states+1)), n_states, clip=True)

    fig, ax=plt.subplots(1,1,figsize=(6, 6))
    s = ax.scatter(c_centers[:,dim_1], c_centers[:,dim_2], 
                   c=state_assignment, s=c_centers_s, cmap=cmap, norm=norm)
    ax.set_xlabel(f'tIC {dim_1+1}', fontsize=14)
    ax.set_ylabel(f'tIC {dim_2+1}', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    cbar_ax = fig.add_axes([ax.get_position().x1+0.02, ax.get_position().y0, 0.03, ax.get_position().height])
    cbar = plt.colorbar(s, cax=cbar_ax, label='Macrostate')
    cbar.set_ticks(np.arange(n_states) + 0.5)
    cbar.set_ticklabels(np.arange(n_states) + 1)
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.set_ylabel('Macrostate', fontsize=14)

    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_mfpt_matrix(mfpt, dt, mfpt_err=None, text_f =".2e", savedir=None):
    """
    Parameters
    ----------
    mfpt : np.ndarray (n_states, n_states)
        Matrix of mean first passage times between states
    dt : float
        The time interval of the feature trajectories in nanoseconds. Should be consistent with 
        dt_out in the study object. 
    mfpt_err : optional, np.ndarray (n_states, n_states) 
        Matrix of standard error of mean first passage times between states
    text_f : optional, str
        Format of the text in the plot        
    savedir : optional, str
        Path to save the plot
    """
    n_states = mfpt.shape[0]
    mfpt_scaled = mfpt*dt/1000
    mfpt_err_scaled = mfpt_err*dt/1000 if mfpt_err is not None else None
    norm = mpl.colors.Normalize(vmin=np.min(mfpt_scaled), vmax=np.max(mfpt_scaled))

    fig,ax = plt.subplots(1,figsize=(10,10))
    s = ax.imshow(mfpt_scaled, cmap='turbo', norm=norm) 
    for i in range(n_states):
        for j in range(n_states):
            if mfpt_err_scaled is not None and i != j:
                ax.text(j, i, "{value:{format}} \n$\pm$ {err:{format}}".format(value=mfpt_scaled[i,j], err=mfpt_err_scaled[i,j], format=text_f), va='center', ha='center', color='white', size=10, weight="bold")
            else:
                ax.text(j, i, "{value:{format}}".format(value=mfpt_scaled[i,j], format=text_f), va='center', ha='center', color='white', size=10, weight="bold")

    cbar_ax = fig.add_axes([ax.get_position().x1+0.02, ax.get_position().y0, 0.03, ax.get_position().height])
    cbar = plt.colorbar(s, cax=cbar_ax)

    ax.set_xlabel('State$_j$', fontsize=18)
    ax.set_ylabel('State$_i$', fontsize=18)
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(np.arange(n_states)+1, fontsize=14)
    ax.set_yticklabels(np.arange(n_states)+1, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_title('MFPT Matrix', fontsize=25)

    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel(r"State$_i$ --> State$_j$ MFPT ($\mathrm{\mu s}$)", fontsize=16)

    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None


def plot_dihed_pie(spatial_counts, dihed_counts, 
                   show_legend=False, show_dihed='all', radius_size=0.5, 
                   dfg_spatial_colors = dfg_spatial_colors,
                   dfg_dihed_colors = dfg_dihed_colors,
                   figsize=(6,6), title='', fontsize=12, savedir=None):
    
    spatial_cluster_labels = ['noise', 'DFG-in', 'DFG-inter', 'DFG-out']
    dihed_cluster_labels = ['noise', 
                            'noise', 'BLAminus', 'BLAplus', 'ABAminus', 'BLBminus', 'BLBplus', 'BLBtrans', 
                            'noise', 'BABtrans', 
                            'noise', 'BBAminus']
    
    inner_colors = dfg_spatial_colors
    outer_colors = [dfg_dihed_colors[0],     # white for undefined spatial group
                    dfg_dihed_colors[0], *dfg_dihed_colors[1:7],  # DFG-in sub-clusters
                    dfg_dihed_colors[0], dfg_dihed_colors[7],  # DFG-inter sub-clusters
                    dfg_dihed_colors[0], dfg_dihed_colors[8]]  # DFG-out sub-clusters
    
    filtered_spatial_cluster_labels = [label if count/sum(spatial_counts) >= 0.05 else '' for label, count in zip(spatial_cluster_labels, spatial_counts)]
    filtered_dihed_cluster_labels = [label if count/sum(sum(dihed_counts,[])) >= 0.05 else '' for label, count in zip(dihed_cluster_labels, sum(dihed_counts,[]))]

    fig, ax = plt.subplots(figsize=figsize)
 

    if show_dihed == 'all':
        wedges_i, texts_i = ax.pie(spatial_counts, radius=1-radius_size, 
                                colors=inner_colors, 
                                wedgeprops=dict(width=radius_size, edgecolor='w'))        
        wedges_o, texts_o = ax.pie(sum(dihed_counts,[]), radius=1, 
                                   colors=outer_colors,
                                   labels=filtered_dihed_cluster_labels, 
                                   textprops={'fontsize': fontsize, 'fontweight': 'bold'}, 
                                   labeldistance=0.7,
                                   wedgeprops=dict(width=radius_size, edgecolor='w'))
    elif show_dihed == 'no_labels':
        wedges_i, texts_i = ax.pie(spatial_counts, radius=1-radius_size, 
                        colors=inner_colors, 
                        wedgeprops=dict(width=radius_size, edgecolor='w'))   
        wedges_o, texts_o = ax.pie(sum(dihed_counts,[]), radius=1, 
                                   colors=outer_colors,
                                   wedgeprops=dict(width=radius_size, edgecolor='w'))
    else:
        wedges_i, texts_i = ax.pie(spatial_counts, radius=0.6, 
                                    colors=inner_colors, 
                                    wedgeprops=dict(width=radius_size, edgecolor='w'))
        
    ax.set(aspect="equal")
    ax.text(0, 0, title, ha='center', va='center', fontsize=16, fontweight='bold')

    if show_legend:
        ax.legend(wedges_i, spatial_cluster_labels,
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=18,
                markerscale=2)
    
    if savedir is not None:
        plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None 