# -*- coding: utf-8 -*-
"""
Script for producing Figure 2. 
Rank correlations in South African and Finnish National parks between social media and official statistics.

Code associated to following article:
    
    Henrikki Tenkanen, Enrico Di Minin, Vuokko Heikinheimo, Anna Hausmann, Marna Herbst, Liisa Kajala & Tuuli Toivonen. (2017).
    Instagram, Flickr, or Twitter: Assessing the usability of social media data for visitor monitoring in protected areas. 
    Scientific Reports 7, 17615. doi:10.1038/s41598-017-18007-4

Author: 
    Henrikki Tenkanen, Digital Geography Lab, Department of Geosciences and Geography, University of Helsinki.

Requirements:
    geopandas
    geoplot
    pandas
    matplotlib
    
Created on:
    Sun May 21 17:00:06 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/

"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def rankOrder(df, rank_col, target_col):
    """
    Determine rank order of rank column into target column.
    
    Parameters
    ----------
    
    df : pd.DataFrame
        Pandas DataFrame containing the data
    rank_col : str
        Column name of the column that will be used to determine the ranking.
    target_col : str
        Target column for the ranks.
        
    Returns
    -------
    pd.DataFrame
    """
    # Make index col
    df['index'] = df.index
    
    # Take copy
    dfc = df.copy()
    
    # Sort data
    dfc = dfc.sort_values(by=rank_col, ascending=False)
    
    # Reset index
    dfc = dfc.reset_index()
    
    # Determine rank
    dfc[target_col] = dfc.index + 1 
    
    # Join back
    df = df.merge(dfc[['index', target_col]], on='index')
    
    # Drop index column
    df = df.drop('index', axis=1)
    
    return df

def adjustGridlines(ax, grid_linewidth=0.5):
    """Helper function to adjust the aesthetics of the gridlines"""
    gridlines = ax.get_xgridlines()
    gridlines.extend( ax.get_ygridlines() )

    for line in gridlines:
                line.set_linewidth(grid_linewidth)
                line.set_linestyle('dotted')
    ax.grid(True)
    return ax

def commonAxisLabels(fig, xlabel=None, ylabel=None, labelpad=12, props=None):
    """
    Add common x and y labels for subplots.
    
    Parameters
    ----------
    fig : plt.Figure()
        Figure object containing the subplots.
    xlabel : str
        Label text for x-axis.
    ylabel : str
        Label text for y-axis.
    props : dict
        Possible text properties for labels.    
    
    Returns
    -------
    plt.Figure
    
    """
    # Add common ax labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid('off')
    if xlabel:
        plt.xlabel(xlabel, labelpad=12, **props)
    if ylabel:
        plt.ylabel(ylabel, labelpad=12, **props)
    return fig

def plotRanks(ax, df, xcol, ycol, color='blue', perf_line=None, N=None, xlabel='', ylabel='', title='', show_info=False, text='', text_x=None, text_y=None, arrow_sx=None, arrow_sy=None, arrow_ex=None, arrow_ey=None, spearman_r=None, sx=None, sy=None):
    """
    Helper function to plot rank correlation plot. 
    """
    
    # Plot line of perfect rank order
    ax.plot(perf_line, perf_line, color='k', linewidth=0.5)
    
    # Plot ranks
    df.plot(ax=ax, x=xcol, y=ycol, kind='scatter', color=color, s=18)
    
    # Limits
    ax.set_xlim(0, N+1); ax.set_ylim(0, N+1)
    
    # Remove axis labels
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    
    # Adjust gridlines
    ax = adjustGridlines(ax)
     
    # Add titles
    ax.set_title(title, family='Arial', fontsize=16)
    
    # Add Spearman correlations
    ax.text(sx, sy, "$r_s$: %0.2f" % spearman_r, fontsize=15, family='Arial') 

    # Add arrows and info texts
    if show_info:
        ax.text(text_x, text_y, text, rotation=40, ha='center', fontsize=10, color='gray', family='Arial')
        ax.annotate('', xy=(arrow_sx, arrow_sy), xycoords='data', 
                     xytext=(arrow_ex, arrow_ey), textcoords='data',
                     arrowprops=dict(arrowstyle="<-", color='gray'))
    
    # Add ycolumn ticks only for the first subplots (from the left)
    if not show_info:
        ax.yaxis.set_ticklabels([])
    
    return ax


# Use White-grid style
plt.style.use('seaborn-whitegrid')

# Filepaths
sa_fp = "SANParks_Visitor_Statistics_and_UserDays_2014.csv"
fi_fp = "Finland_Visitor_Statistics_and_UserDays_2014.csv"
outfp = "SA_FI_Park_Spearman_rankings.png"

# Read files
sa = pd.read_csv(sa_fp, sep=';')
fi = pd.read_csv(fi_fp, sep=';', encoding='latin1')

# Drop parks that does not have official stats
sa = sa.ix[sa['Off_visitors']> 0]
fi = fi.ix[fi['Off_visitors']> 0]

# Determine ranks for different parks according 
# social media user-days and official statistics
# ----------------------------------------------

# South Africa
# ------------
sa = rankOrder(sa, rank_col='Off_visitors', target_col='O_rank')
sa = rankOrder(sa, rank_col='SUD', target_col='S_rank')
sa = rankOrder(sa, rank_col='IUD', target_col='I_rank')
sa = rankOrder(sa, rank_col='TUD', target_col='T_rank')
sa = rankOrder(sa, rank_col='FUD', target_col='F_rank')

# Finland
# -------
fi = rankOrder(fi, rank_col='Off_visitors', target_col='O_rank')
fi = rankOrder(fi, rank_col='SUD', target_col='S_rank')
fi = rankOrder(fi, rank_col='IUD', target_col='I_rank')
fi = rankOrder(fi, rank_col='TUD', target_col='T_rank')
fi = rankOrder(fi, rank_col='FUD', target_col='F_rank')

# Calculate rank correlations
# ---------------------------

# SA
# --
sa_sud_r, sa_sud_p = spearmanr(sa['Off_visitors'], sa['SUD'])
sa_iud_r, sa_iud_p = spearmanr(sa['Off_visitors'], sa['IUD'])
sa_tud_r, sa_tud_p = spearmanr(sa['Off_visitors'], sa['TUD'])
sa_fud_r, sa_fud_p = spearmanr(sa['Off_visitors'], sa['FUD'])

# FI
# --

fi_sud_r, fi_sud_p = spearmanr(fi['Off_visitors'], fi['SUD'])
fi_iud_r, fi_iud_p = spearmanr(fi['Off_visitors'], fi['IUD'])
fi_tud_r, fi_tud_p = spearmanr(fi['Off_visitors'], fi['TUD'])
fi_fud_r, fi_fud_p = spearmanr(fi['Off_visitors'], fi['FUD'])

# Straight line values (perfect rank order)
# -----------------------------------------
sa_N = len(sa)
fi_N = len(fi)

# Line values
sa_line = [x + 1 for x in range(sa_N)]
fi_line = [x + 1 for x in range(fi_N)]


# Visualize rank orders
# ---------------------

# Plot using subplots with 2 rows and 4 columns
nrows = 2
ncolumns = 4
fig, axarray = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(12,6))

# Columns (x and y pairs)
cols = [['O_rank', 'S_rank'],
        ['O_rank', 'I_rank'],
        ['O_rank', 'T_rank'],
        ['O_rank', 'F_rank'],
         ]

# Titles
titles = ['All platforms', 'Instagram', 'Twitter', 'Flickr']

# Spearman correlation values with different platforms
sa_spearmans = [sa_sud_r, sa_iud_r, sa_tud_r, sa_fud_r]
fi_spearmans = [fi_sud_r, fi_iud_r, fi_tud_r, fi_fud_r]

rowidx = 0
colidx = 0

# Plot data 
for idx in range(nrows*ncolumns):
    # Get ax
    ax = axarray[rowidx][colidx]

    # Plot South Africa ranks
    # -----------------------
    if rowidx == 0:
        
        # Plot parameters
        data = sa
        xcol, ycol = cols[colidx][0], cols[colidx][1]
        N = sa_N
        perfect_line_values = sa_line
        color = 'blue'
        title = titles[colidx]
        infotext = "Social media\nunderestimates\npopularity"
        text_x, text_y = 4.0, 18.2
        arrow_start_x, arrow_end_x = 2.35, 0.0
        arrow_start_y, arrow_end_y = 19.0, 22.0
        spearman_value = sa_spearmans[colidx]
        spearman_x, spearman_y = 15, 1.5
        show_info = False
        
        # Show plot guide only in first column
        if colidx == 0:
            show_info = True
                    
        ax = plotRanks(ax=ax, df=data, xcol=xcol, ycol=ycol, color=color, 
                       perf_line=perfect_line_values, N=N, 
                       title=title, 
                       show_info=show_info, text=infotext, text_x=text_x, text_y=text_y, 
                       arrow_sx=arrow_start_x, arrow_sy=arrow_start_y, arrow_ex=arrow_end_x, arrow_ey=arrow_end_y,
                       spearman_r=spearman_value, sx=spearman_x, sy=spearman_y)
        
        # Add y label for first subplot
        if colidx == 3:
            ylabel = 'South Africa'
            ax.set_ylabel(ylabel, labelpad=13, **{'fontsize': '16', 'family': 'Arial', 'rotation': '270'})
            # Set the title on the right side
            ax.yaxis.set_label_position("right")
        
    # Plot Finnish ranks
    # ------------------
    elif rowidx==1:
        # Plot parameters
        data = fi
        xcol, ycol = cols[colidx][0], cols[colidx][1]
        N = fi_N
        perfect_line_values = fi_line
        color = 'red'
        title = ''
        infotext = "Social media\noverestimates\npopularity"
        text_x, text_y = 30.5, 8.0
        arrow_start_x, arrow_end_x = 32.5, 36.0
        arrow_start_y, arrow_end_y = 4.8, 0.0
        spearman_value = fi_spearmans[colidx]
        spearman_x, spearman_y = 1, 31.5
        show_info = False
        
        if colidx == 0:
            show_info = True
        
        ax = plotRanks(ax=ax, df=data, xcol=xcol, ycol=ycol, color=color, 
                       perf_line=perfect_line_values, N=N, 
                       title=title, 
                       show_info=show_info, text=infotext, text_x=text_x, text_y=text_y, 
                       arrow_sx=arrow_start_x, arrow_sy=arrow_start_y, arrow_ex=arrow_end_x, arrow_ey=arrow_end_y,
                       spearman_r=spearman_value, sx=spearman_x, sy=spearman_y)
        if colidx == 3:
            ylabel = 'Finland'
            ax.set_ylabel(ylabel, labelpad=13, **{'fontsize': '16', 'family': 'Arial', 'rotation': '270'})
            # Set the title on the right side
            ax.yaxis.set_label_position("right")
            
            # Add info arrow for the line of perfect rank order
            infotext = "Line of perfect match"
            text_x, text_y = 14, 3
            arrow_start_x, arrow_end_x = 14.0, 11.0
            arrow_start_y, arrow_end_y = 4.5, 9.5
            
            ax.text(text_x, text_y, infotext, rotation=0, ha='left', fontsize=10, color='gray', family='Arial')
            ax.annotate('', xy=(arrow_start_x, arrow_start_y), xycoords='data', 
                     xytext=(arrow_end_x, arrow_end_y), textcoords='data',
                     arrowprops=dict(arrowstyle="<-", color='gray'))
        
    # Update indices
    colidx+=1
    if colidx == 4:
        rowidx+=1
        colidx=0
        
# Add common labels
xlabel = "Official statistics - Park popularity ranking"
ylabel = "Social media - Park popularity ranking"
lab_props = {'fontsize': '20', 'family': 'Arial'}
fig = commonAxisLabels(fig=fig, xlabel=xlabel, ylabel=ylabel, labelpad=28, props=lab_props)

# Adjust the margins
plt.subplots_adjust(bottom=0.12, top=0.95, left=0.075, right=0.97, wspace=0.02, hspace=0.15)

# Save to disk
plt.savefig(outfp, dpi=600)

