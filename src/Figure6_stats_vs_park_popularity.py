# -*- coding: utf-8 -*-
"""
Script for producing Figure 6. 
Plot a graph comparing the Pearson correlations to Visitor rates | Density of social media user-days.

Code associated to following article:
    
    Henrikki Tenkanen, Enrico Di Minin, Vuokko Heikinheimo, Anna Hausmann, Marna Herbst, Liisa Kajala & Tuuli Toivonen. (2017).
    Instagram, Flickr, or Twitter: Assessing the usability of social media data for visitor monitoring in protected areas. 
    Scientific Reports 7, 17615. doi:10.1038/s41598-017-18007-4

Author: 
    Henrikki Tenkanen, Digital Geography Lab, Department of Geosciences and Geography, University of Helsinki.

Requirements:
    pandas
    numpy
    math
    matplotlib
    scipy
    
Created on:
    Fri Apr 21 09:33:03 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/                                       
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import math, os
import numpy as np
import scipy.stats as stats

def fitLine(df_data, x_col, y_col, alpha=0.05, plotFlag=1, ax=None, outfp=None, ylabel=None, show_legend=True):
    """ 
    Fit a curve to the data using a least squares 1st order polynomial fit.
    
    Source
    ------
    
    Modified after: https://github.com/thomas-haslwanter/statsintro_python/tree/master/ISP/Code_Quantlets/11_LinearModels/fitLine
    
    Info
    ----
    http://reliawiki.org/index.php/Simple_Linear_Regression_Analysis
    
    """
    
    # Get ax and y as numpy arrays
    x, y = df_data[x_col].values, df_data[y_col].values
    
    # Summary data
    n = len(x)			   # number of samples     
    
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Linefit
    b = Sxy/Sxx
    a = mean_y - b*mean_x
    
    # Residuals
    fit = lambda xx: a + b*xx    
    residuals = y - fit(x)
    
    var_res = np.sum(residuals**2)/(n-2)
    sd_res = np.sqrt(var_res)
    
    # Confidence intervals
    se_b = sd_res/np.sqrt(Sxx)
    se_a = sd_res*np.sqrt(np.sum(x**2)/(n*Sxx))
    
    df = n-2                            # degrees of freedom
    tval = stats.t.isf(alpha/2., df) 	# appropriate t value
    
    # Confidence interval values
    ci_a = a + tval*se_a*np.array([-1,1])
    ci_b = b + tval*se_b*np.array([-1,1])

    # create series of new test x-values to predict for
    npts = 100
    px = np.linspace(np.min(x),np.max(x),num=npts)
    
    se_fit     = lambda x: sd_res * np.sqrt(  1./n + (x-mean_x)**2/Sxx)
    se_predict = lambda x: sd_res * np.sqrt(1+1./n + (x-mean_x)**2/Sxx)
    
    # Predicted values
    pred_vals = se_predict(x)    
        
    # Return info
    ri = {'residuals': residuals, 
        'var_res': var_res,
        'sd_res': sd_res,
        'pred_vals': pred_vals,
        'alpha': alpha,
        'tval': tval,
        'df': df}
    
    if plotFlag == 1:
        
        if show_legend:
            ax.plot(px, fit(px),'k', label='Regression line')
        else:
            ax.plot(px, fit(px),'k')
        
        x.sort()
        limit = (1-alpha)*100
                
        conflabel='Confidence limit ({0:.1f}%)'.format(limit)
        predictlabel='Prediction limit ({0:.1f}%)'.format(limit)
        
        if show_legend:
            ax.plot(x, fit(x)+tval*se_fit(x), 'r--', linewidth=2, label=conflabel)
        else:
            ax.plot(x, fit(x)+tval*se_fit(x), 'r--', linewidth=2)
        ax.plot(x, fit(x)-tval*se_fit(x), 'r--', linewidth=2)
        
        if show_legend:
            ax.plot(x, fit(x)+tval*se_predict(x), 'c--', linewidth=2, label=predictlabel)
        else:
            ax.plot(x, fit(x)+tval*se_predict(x), 'c--', linewidth=2)
        ax.plot(x, fit(x)-tval*se_predict(x), 'c--', linewidth=2)
    
    return ax, conflabel, predictlabel, (a,b,(ci_a, ci_b), ri)
    
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
    
def mostVisitedPark(grouped, visitor_col='N. off. visits'):
    """ Find out what is the most visited park (to normalize the data into a scale 0.0 - 1.0)"""
    most_visited_cnt, most_visited_park = 0, None
    for idx, rows in grouped:
        # Sum the Guests
        guest_sum = rows[visitor_col].sum()
        # Update if it is bigger than earlier parks
        if guest_sum > most_visited_cnt:
            most_visited_cnt = guest_sum
            most_visited_park = idx
    return most_visited_cnt, most_visited_park

def plotPoint(park_group, measure, stat_col, use_log=False, ax=None, point_color='red', point_label=None, result_df=None, annotate=False, ylabel=None):
    """
    Helper function to make a scatter plot based on Pearson correlations and the statistic variable (lg social media | lg official visitors).
    """
    
    # Iterate over parks and plot
    for idx, rows in park_group:
        
        # The goodness measure to be plotted
        goodness = rows[measure].unique()[0]   
        try:
            # Take logarithm of park visitor sum 
            if use_log:
                park_stat = math.log(rows[stat_col].sum()) #/ sa_lg_max_vis
            else:
                park_stat = rows[stat_col].sum()
        except:
            park_stat = 0
            
        # DF
        df = pd.DataFrame([[goodness, park_stat]])
        df.columns = ['goodness', 'park_stat']
        
        # Plot X and Y
        df.plot(x='goodness', y='park_stat', kind='scatter', s=18, ax=ax, color=point_color)
        
        if annotate:
            # Set Park-ID as index
            df.index = df['park_id']        
            
            # Annotate the values
            for k, v in df[['goodness', 'park_stat']].iterrows():
                ax.annotate(k, v, fontsize=6, textcoords='offset points')   
        
        # Add to DataFrame
        result_df = result_df.append(df)
            
    return ax, result_df

def plotCross(ax, df):
    """ Plot vertical and horizontal lines to make a 'cross' (optional) """
    #ymin, ymax = 7.5, 15.5 #0.3, 1.3
    ymin, ymax = df['park_stat'].min(), df['park_stat'].max() #0.3, 1.3
    ymean = ymax-((ymax-ymin)/2)
    ax.plot((0.5, 0.5), (ymin, ymax), c='black', lw=0.3)
    ax.plot((0.0, 1.0), (ymean, ymean), c='black', lw=0.3)
    return ax

def visualize(ax, df, xcolumn, ycolumn, xlabel, ylabel, ylims, annotate, point_colors=['blue'], point_labels=None, plot_cross=False, show_legend=True):
    """
    Helper function to visualize the scatter plots and calculate/plot basic statistics.
    """

    # Result df 
    points = pd.DataFrame()
    
    # Plot points
    for idx, data in enumerate(df):
        ax, points = plotPoint(park_group=data, measure=xcolumn, stat_col=ycolumn, use_log=True, ax=ax, point_color=point_colors[idx], point_label=point_labels[idx], result_df=points, annotate=annotate)
        
    # Plot cross
    if plot_cross:
        ax = plotCross(ax, points)

    # Do a simple linear regression to get the slope etc.
    slope, intercept, pearson_corr, pvalue, stderr = stats.linregress(points['park_stat'], points['goodness'])    
    print("\nSlope: %.2f Intercept: %.2f Pearson: %.2f P-value: %.2f Std: %.2f\n" % (slope, intercept, pearson_corr, pvalue, stderr))
 
    # Fit trend line (least squares 1st order Polynomial fit)
    # --------------------------------------------------------
    ax, conflabel, predictlabel, (a,b,(ci_a, ci_b), ri,newy) = fitLine(df_data=points, x_col='goodness', y_col='park_stat', alpha=0.01,newx=np.array([1,4.5]), ax=ax, outfp=outfp, ylabel=ylabel, show_legend=show_legend)  
        
    # Set labels
    lab_props = {'fontsize': '14', 'family': 'Arial', "labelpad" : 10.0}
    ax.set_ylabel(ylabel, **lab_props)
    ax.set_xlabel(xlabel, **lab_props)
    
    # Set value limits
    ax.set_xlim(-0.475, 1.1)
    ax.set_ylim(ylims[0], ylims[1])
        
    # Set Pearson correlation and p-value to ri
    ri['pearson_r'] = pearson_corr
    ri['pearson_r_pvalue'] = pvalue
    ri['slope'] = slope
    ri['intercept'] = intercept
    ri['std_error'] = stderr
    
    # Adjust gridlines
    ax = adjustGridlines(ax)

    return ax, points, ri

def graph2x4(datasets, measure, stat_col, y_labels, colors, plabels):
    """
    Helper function to produce subplots with 2 rows and 4 columns to investigate the relation 
    between social media user-days and the popularity of the park.
    """
    # Initialize figure and axes
    nrows = 2
    ncolumns = 4
    fig, axarray = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(11,5))

    rowidx = 0
    colidx = 0
    
    # Plot data 
    for idx in range(nrows*ncolumns):
        # Get ax
        ax = axarray[rowidx][colidx]
        
        # Show legend
        show_legend = False
        
        # Show the legend in the second axis
        if rowidx == 1:
            if colidx == 0:
                show_legend=True
            
        # Xcolumn
        xcol = measure[colidx]
        
        ycolumn = stat_col[colidx][rowidx]
        print("============================\nX: ", xcol, "Y: ", ycolumn,"\n============================")
        
        # Define xlabel
        if rowidx == 0:
            title = y_labels[colidx]
            xlabel = ''
            ylims = (4, 16)
            if colidx == 0:
                ylabel = 'lg Official visitors'
            else:
                ylabel = ''
            
            # Regression formula position
            reg_xpos = -0.45
            reg_ypos = 15.25

        elif rowidx == 1:
            title = None
            xlabel = ''
            ylims = (-5, 11)
            if colidx == 0:
                ylabel = 'lg Social media'
            else:
                ylabel = ''
            
            # Regression formula position
            reg_xpos = -0.45
            reg_ypos = 10.1
            
        # Visualize
        ax, points, ri = visualize(ax=ax, df=data_sets, xcolumn=xcol, ycolumn=ycolumn, xlabel=xlabel, ylabel=ylabel, ylims=ylims, point_colors=colors, point_labels=plabels, plot_cross=False, annotate=False, show_legend=show_legend)
        
        # Add title
        if title:
            ax.set_title(title, family='Arial', fontsize=16)
            
        # Use y-ticklabels only in the first subplots (from leftside)
        if not colidx == 0:
            ax.yaxis.set_ticklabels([])
        
        # Use x-ticklabels only in the bottom row
        if not rowidx == 1:
            ax.xaxis.set_ticklabels([])
        
        # Parse correlation info text
        r = ri['pearson_r']
        pval = np.round(ri['pearson_r_pvalue'], 2)
        if pval <= 0.01:
            corr_info = "r = %.2f***" % r
        elif pval <= 0.05:
            corr_info = "r = %.2f**" % r
        elif pval <= 0.1:
            corr_info = "r = %.2f*" % r
        else:
            corr_info = "r = %.2f" % r
        
        # Do not put the whole formula because it is confusing
        reg_formula = "%s,  slope = %.2f" % (corr_info, ri['slope'])
        ax.text(reg_xpos, reg_ypos, reg_formula, **{'fontsize': '8', 'family': 'Arial'})
        
        # Set legend
        if show_legend:
            # Line legend
            legend = ax.legend(loc='lower right', shadow=True, fontsize=7)
            legend.get_frame().set_facecolor('#00FFCC')
            
            # Color legend
            xpos = -0.2
            ypos = -2.7
            ax.plot(xpos, ypos, marker='o', markersize=5, color='red')
            ax.plot(xpos, ypos-1, marker='o', markersize=5, color='blue')
            ax.text(xpos+0.07, ypos-0.3, "FI", fontsize=8)
            ax.text(xpos+0.07, ypos-1.4, "SA", fontsize=8)
            
            # Adjust the ytick labels
            ax.yaxis.set_ticks(np.arange(0,10.1, 2))
            
        # Set xaxis ticks
        ax.xaxis.set_ticks(np.arange(-0.25,1.1, 0.25))
        
        # Add legend about p-values
        if rowidx==1 and colidx==1:
            infotxt = "***: p-value < 0.01\n**: p-value < 0.05"
            ax.text(0.51,-4.6, infotxt, **{'fontsize': '8', 'family': 'Arial'})
        
        # Update indices
        colidx+=1
        if colidx==4:
            rowidx+=1
            colidx=0
    
    return fig, ax, points, ri


# Filepaths
# ---------
sa_stats_fp = "SANParks_stats_comparison_OfficialInstagramTwitterFlickr_2014.csv"
sa_numbers_fp = "SANParks_Visitor_Statistics_and_UserDays_2014.csv"
fi_stats_fp = "Finland_stats_comparison_OfficialInstagramTwitterFlickr_2014.csv"
fi_numbers_fp = "Finland_Visitor_Statistics_and_UserDays_2014.csv"
outfolder = "/home/results"

# Parameters
# ==========

fin_name_convert = {'Etelä-Konneveden kansallispuisto': 'Etelä-Konnevesi',
                'Helvetinjärven kansallispuisto': 'Helvetinjärvi',
                'Hiidenportin kansallispuisto': 'Hiidenportti',
                'Isojärven kansallispuisto': 'Isojärvi',
                'Itäisen Suomenlahden kansallispuisto': 'Itäinen Suomenlahti',
                'Kauhanevan-Pohjankankaan kansallispuisto': 'Kauhaneva-Pohjankangas',
                'Kolin kansallispuisto': 'Koli',
                'Koloveden kansallispuisto': 'Kolovesi',
                'Kurjenrahkan kansallispuisto': 'Kurjenrahka',
                'Lauhanvuoren kansallispuisto': 'Lauhanvuori',
                'Leivonmäen kansallispuisto': 'Leivonmäki',
                'Lemmenjoen kansallispuisto': 'Lemmenjoki',
                'Liesjärven kansallispuisto': 'Liesjärvi',
                'Nuuksion kansallispuisto': 'Nuuksio',
                'Oulangan kansallispuisto': 'Oulanka',
                'Pallas-Yllästunturin kansallispuisto': 'Pallas-Yllästunturi',
                'Patvinsuon kansallispuisto': 'Patvinsuo',
                'Petkeljärven kansallispuisto': 'Petkeljärvi',
                'Puurijärven ja Isonsuon kansallispuisto': 'Puurijärvi ja Isosuo',
                'Pyhä-Häkin kansallispuisto': 'Pyhä-Häkki',
                'Pyhä-Luoston kansallispuisto': 'Pyhä-Luosto',
                'Päijänteen kansallispuisto': 'Päijänne',
                'Repoveden kansallispuisto': 'Repovesi',
                'Riisitunturin kansallispuisto': 'Riisitunturi',
                'Rokuan kansallispuisto': 'Rokua',
                'Saaristomeren kansallispuisto': 'Saaristomeri',
                'Salamajärven kansallispuisto': 'Salamajärvi',
                'Seitsemisen kansallispuisto': 'Seitseminen',
                'Selkämeren kansallispuisto': 'Selkämeri',
                'Sipoonkorven kansallispuisto': 'Sipoonkorpi',
                'Syötteen kansallispuisto': 'Syöte',
                'Tiilikkajärven kansallispuisto': 'Tiilikkajärvi',
                'Torronsuon kansallispuisto': 'Torronsuo',
                'Urho Kekkosen kansallispuisto': 'Urho Kekkonen',
                'Valkmusan kansallispuisto': 'Valkmusa'}


# Use Seaborn White grid style
plt.style.use('seaborn-whitegrid')

# Annotate points
annotate = False

# Parks with temporal Autocorrelation
autocor_parks = ['Camdeboo', 'Tsitsikamma', 'Mokala', 
                 'Helvetinjärvi', 'Isojärvi','Lauhanvuori', 
                 'Leivonmäki', 'Lemmenjoki', 'Liesjärvi', 'Nuuksio', 
                 'Oulanka', 'Patvinsuo', 'Petkeljärvi', 'Pyhä-Häkki', 
                 'Päijänne', 'Repovesi', 'Rokua', 'Salamajärvi', 
                 'Seitseminen','Selkämeri']

# Read files 
# ----------

# SA
sa_data = pd.read_csv(sa_stats_fp, sep=';')
sa_counts = pd.read_csv(sa_numbers_fp, sep=';')

# FI
fi_data = pd.read_csv(fi_stats_fp, sep=';', encoding='latin1')
fi_counts = pd.read_csv(fi_numbers_fp, sep=';', encoding='latin1')

# Join Park size into the statistics layers
sa_data = sa_data.merge(sa_counts, on='Park')
fi_data = fi_data.merge(fi_counts, on='Park')

# Remove NoData #, 'I_PCORR', 'T_PCORR', 'F_PCORR'
sa_data = sa_data.dropna(subset=['Off_visitors', 'S_PCORR'])
fi_data = fi_data.dropna(subset=['Off_visitors', 'S_PCORR'])

# Fill other with 0
sa_data = sa_data.fillna(0)
fi_data = fi_data.fillna(0)

# Remove parks with temporal autocorrelation
# ------------------------------------------
sa_data = sa_data.ix[~sa_data['Park'].isin(autocor_parks)]
fi_data = fi_data.ix[~fi_data['Park'].isin(autocor_parks)]

# Group by park
# -------------
sa_grouped = sa_data.groupby('Park')
fi_grouped = fi_data.groupby('Park')

# Find out what is the most visited park (to normalize the data into a scale 0.0 - 1.0)
# --------------------------------------------------------------------------------------

# Column 
offcol = 'Off_visitors'

# SA
sa_most_visited_cnt, sa_most_visited_park = mostVisitedPark(sa_grouped, visitor_col=offcol)

# FI
fi_most_visited_cnt, fi_most_visited_park = mostVisitedPark(fi_grouped, visitor_col=offcol)

# Take log from Most visited park
# -------------------------------

# SA
sa_lg_max_vis = math.log(sa_most_visited_cnt)

# FI
fi_lg_max_vis = math.log(fi_most_visited_cnt)

# Data that will be used (needs to be in a list)
# ---------------------------------------------
data_sets = [sa_grouped, fi_grouped]

# The measure to be plotted:
# ---------------------------
measure = ['S_PCORR',
           'I_PCORR',
           'T_PCORR',
           'F_PCORR',
           ] 

# Statistics that will be used in plotting
stat_col = [[offcol, 'SUD'],
            [offcol, 'IUD'],
            [offcol, 'TUD'],
            [offcol, 'FUD']]

y_labels = ['All platforms', 
            'Instagram', 
            'Twitter', 
            'Flickr']

# Colors for the datasets
colors = ['blue', 'red']

# Point labels
plabels = ['SA', 'FI']

# ===============
# Plot the figure
# ===============

fig, ax, points, ri = graph2x4(datasets=data_sets, measure=measure, stat_col=stat_col, y_labels=y_labels, colors=colors, plabels=plabels)

# Plot legend for points 
# ======================

# Add common labels
xlabel = "Pearson's correlation"
ylabel = ""
lab_props = {'fontsize': '20', 'family': 'Arial'}
fig = commonAxisLabels(fig=fig, xlabel=xlabel, ylabel=ylabel, labelpad=24, props=lab_props)

# Adjust the margins
plt.subplots_adjust(bottom=0.13, top=0.95, left=0.065, right=0.99, wspace=0.02, hspace=0.04)

# Save to disk
outfp = os.path.join(outfolder, "SAFI_NPs_Some_vs_Visitors_and_SUD.jpg")
plt.savefig(outfp, dpi=400)