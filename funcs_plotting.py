# Functions for plotting MSM results

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from deeptime.plots import plot_implied_timescales, plot_energy2d, plot_contour2d_from_xyz
from deeptime.util import energy2d

import numpy as np


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

    plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None



def plot_fe(traj_all, traj_weights, savedir, cmap='nipy_spectral', plot_c_centers=True, \
            dim_1 = 0, dim_2 = 1, \
            c_centers=None, c_centers_s=3, c_centers_marker='.', c_centers_a=0.5, c_centers_c='black'):
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax, contour, cbar = plot_energy2d(energy2d(traj_all[:, dim_1], traj_all[:, dim_2], weights=traj_weights), ax=ax, contourf_kws=dict(cmap=cmap))
    if plot_c_centers: 
        ax.scatter(c_centers[:,dim_1], c_centers[:,dim_2], s=c_centers_s, c=c_centers_c, marker=c_centers_marker, alpha=c_centers_a)
    ax.set_xlabel(f'tIC {dim_1+1}', fontsize=14)
    ax.set_ylabel(f'tIC {dim_2+1}', fontsize=14)
    cbar.ax.set_ylabel('Free energy (kT)', fontsize=14)

    plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None



def plot_ts(timescales, n_ts, markov_lag, savedir, scaling=0.001, unit="$\mathrm{\mu s}$"):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_ts)+2
    y = timescales[:n_ts]*scaling
    markov_lag_scaled = markov_lag * scaling

    ax.plot(x, y, marker='o', markersize=10)
    ax.fill_between([0, n_ts+2], y1=markov_lag_scaled, y2=0, color='grey', alpha=0.8)

    ax.set_yscale('log')
    ax.grid(visible=True, axis='y')

    ax.set_xlim([1, n_ts+2])
    ax.set_ylim([min(y)/3, max(y)*3])

    ax.set_xticks(np.arange(2, n_ts+2, 1))
    ax.set_ylabel(rf"Timescale ({unit})", fontsize=14)
    ax.set_xlabel(r"Timescale Index", fontsize=14)
    ax.tick_params(bottom=True, top=False, left=True, right=False)

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

    plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None



def plot_mfpt_matrix(mfpt, savedir, scaling=0.001, unit="$\mathrm{\mu s}$", text_f =".2e"):
    n_states = mfpt.shape[0]
    mfpt_scaled = mfpt*scaling
    norm = mpl.colors.Normalize(vmin=np.min(mfpt_scaled), vmax=np.max(mfpt_scaled))

    fig,ax = plt.subplots(1,figsize=(10,10))
    s = ax.imshow(mfpt_scaled, cmap='turbo', norm=norm) 
    for i in range(n_states):
        for j in range(n_states):        
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
    cbar.ax.set_ylabel(rf"State$_i$ --> State$_j$ MFPT ({unit})", fontsize=16)

    plt.savefig(savedir, transparent=True, bbox_inches='tight', dpi=300)
    plt.show()

    return None