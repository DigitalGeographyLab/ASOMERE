# -*- coding: utf-8 -*-
"""
Script for producing Figure 5 and the tests associated with it. 
First, visualize boxplots for comparing different social media platforms (Instagram, Twitter, Flickr) in South Africa and in Finland.
Second, conduct Kruskal-Wallis and Dunn's post hoc test to assess statistical significance of the differences between platforms. 
Kruskall-Wallis is a non-parametric version of ANOVA to test the overall difference between groups, and it was chosen since our data did not meet the assumption of 
homoscedasticity (prerequisite for ANOVA), as the standard deviations of the groups were not all equal. 
Dunn’s post hoc test to assess whether there were significant differences between individual social media platforms 
(null-hypothesis: no difference between groups). 
The reported p-values were adjusted according to the Holm-Sidak measure that was used to control for family-wise 
error rate.

Code associated to following article:
    
    Henrikki Tenkanen, Enrico Di Minin, Vuokko Heikinheimo, Anna Hausmann, Marna Herbst, Liisa Kajala & Tuuli Toivonen. (2017).
    Instagram, Flickr, or Twitter: Assessing the usability of social media data for visitor monitoring in protected areas. 
    Scientific Reports 7, 17615. doi:10.1038/s41598-017-18007-4

Author: 
    Henrikki Tenkanen, Digital Geography Lab, Department of Geosciences and Geography, University of Helsinki.

Requirements:
    pandas
    matplotlib
    scipy
    rpy2 
    numpy
    
Created on:
    Wed May 17 13:04:37 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import anderson, kstest, bartlett, kruskal

# Import R for Python (rpy2 module)
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
# R console
r = robjects.r

def adjustGridlines(ax, grid_linewidth, xlines=True, ylines=True):
    """Helper function to adjust the aesthetics of the gridlines"""
    if xlines:
        gridlines = ax.get_xgridlines()
    if ylines and xlines:
        gridlines.extend( ax.get_ygridlines() )

    for line in gridlines:
                line.set_linewidth(grid_linewidth)
                line.set_linestyle('dotted')
    ax.grid(True)
    return ax

def parseADnormalResults(stats, critical_values, signif_level, numerical=False):
    """
    Parse the results of K-sample Anderson-Darling test based on critical values and ad-test statistic. 
    Returns the result as a text in a human readable manner. If numerical flag is true, returns only significance level as a number. 
    
    Parameters
    ----------
    stats : float
        Normalized k-sample Anderson-Darling test statistic.
    critical_values : array
        The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%.
    signif_level : float
        An approximate significance level at which the null hypothesis for the provided samples can be rejected.
    
    Returns
    -------
    Human-readable text : str 
    
    or 
    
    Numerical significance level: float
    
    Info
    ----
    
    1. http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/andeksam.htm
    2. http://www.hep.caltech.edu/~fcp/statistics/hypothesisTest/PoissonConsistency/ScholzStephens1987.pdf
        
    """
    
    sig_levels = {0: '15 %',
                  1: '10 %',
                  2: '5 %',
                  3: '2.5 %',
                  4: '1 %'}
    
    # Iterate over critical values and inform the significance level
    # The null hypothesis: two random samples come from the same distribution
    # Rejection at significance levels of 25%, 10%, 5%, 2.5% and 1%
    # ==> If test value is smaller than the critical value at significance level, the null hypothesis cannot be rejected.
    for idx, threshold in enumerate(critical_values):
        if isinstance(threshold, str):
            if numerical:
                return np.nan
            interp_txt = "AD-stats: N/A"
        elif stats < threshold:
            if numerical:
                return float(sig_levels[idx].replace(' %', '')) / 100
            interp_txt = "AD-stats:\nSame distrib! sig-l %s.\n" % (sig_levels[idx])
        else:
            if numerical:
                return -1.0
            interp_txt = "AD-stats:\nDiff distrib!\np-value:\n%s" % np.round(signif_level, 3)
    return interp_txt

def adjustBoxStyle(fig, bp, ocolor='black', fcolor='#3881b9', falpha=0.9, owidth=0.75, wfliers=True, wcolor = 'black', wstyle = 'dotted', wwidth = 0.5, walpha = 0.9, ccolor = 'black', cwidth = 0.75, calpha = 0.9, mcolor = 'black', mwidth = 1.2, malpha = 1.0):
    """Helper function to adjust the aesthetics of the boxplot"""
    # Box style
    # ---------
    
    for box in bp['boxes']:
        # change outline color
        box.set( color=ocolor, linewidth=owidth)
        # change fill color
        box.set( facecolor = fcolor , alpha=falpha)
        
    # Whisker style
    # -------------
    
    for whisker in bp['whiskers']:
        whisker.set(color=wcolor, linewidth=wwidth, linestyle=wstyle, alpha=walpha)
        
    # Caps style
    # ----------
    
    # Change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color=ccolor, linewidth=cwidth, alpha=calpha)
        
    # Median style
        # ------------

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color=mcolor, linewidth=mwidth, alpha=malpha)
    
    # Draw the Figure (so that tick labels are shown properly)
    # More discussion here: http://stackoverflow.com/questions/41122923/getting-tick-labels-in-matplotlib
    fig.canvas.draw()
    
    return fig

# ===============
# PARAMETERS
# ===============

# Parks with temporal Autocorrelation
autocor_parks = ['Camdeboo', 'Tsitsikamma', 'Mokala', 
                 'Helvetinjärvi', 'Isojärvi','Lauhanvuori', 
                 'Leivonmäki', 'Lemmenjoki', 'Liesjärvi', 'Nuuksio', 
                 'Oulanka', 'Patvinsuo', 'Petkeljärvi', 'Pyhä-Häkki', 
                 'Päijänne', 'Repovesi', 'Rokua', 'Salamajärvi', 
                 'Seitseminen','Selkämeri']

# Use Seaborn whitegrid style
plt.style.use('seaborn-whitegrid')

# REMOVE AUTOCORRELATED PARKS
remove_autocor = True

# Filepaths
sa_stats_fp = "SANParks_stats_comparison_OfficialInstagramTwitterFlickr_2014.csv"
fi_stats_fp = "Finland_stats_comparison_OfficialInstagramTwitterFlickr_2014.csv"
output_fp = "/home/results/SA_FI_platform_comparisons_boxplot_PearsonCorrelation_2014.jpg"

# Read files
sa = pd.read_csv(sa_stats_fp, sep=';')
fi = pd.read_csv(fi_stats_fp, sep=';', encoding='latin1')

# Remove parks with temporal autocorrelation
if remove_autocor:
    sa = sa.ix[~sa['Park'].isin(autocor_parks)]
    fi = fi.ix[~fi['Park'].isin(autocor_parks)]
else:
    print("WARNING RESULTS WITH AUTOCORRELATION\n----------------------\n")

# Initialize figure and axes for SA national parks and FI national parks
fig, axarray = plt.subplots(nrows=1, ncols=2, figsize=(7.5,3), sharey=True)
sax = axarray[0]
fax = axarray[1]

# Create boxplot
stat_cols = ['S_PCORR', 'I_PCORR', 'T_PCORR', 'F_PCORR']
sa_bp = sa.boxplot(ax=sax, column=stat_cols, vert=True, patch_artist=True, return_type='dict')
fi_bp = fi.boxplot(ax=fax, column=stat_cols, vert=True, patch_artist=True, return_type='dict')

# Use different x ticklabels
xlabels = ['All platforms', 'Instagram', 'Twitter', 'Flickr']
sax.xaxis.set_ticklabels(xlabels, fontsize=10)
fax.xaxis.set_ticklabels(xlabels, fontsize=10)

# Adjust gridlines
sax = adjustGridlines(ax=sax, grid_linewidth=0.2, xlines=True, ylines=False)
fax = adjustGridlines(ax=fax, grid_linewidth=0.2, xlines=True, ylines=False)

# Adjust Boxplot appearance
facecolor = "#C2D8E9"
fig = adjustBoxStyle(fig=fig, bp=sa_bp, fcolor=facecolor, falpha=1.0, mwidth=3, mcolor='red', owidth=1)
fig = adjustBoxStyle(fig=fig, bp=fi_bp, fcolor=facecolor, falpha=1.0, mwidth=3, mcolor='red', owidth=1)

# Add title
lab_props = {'fontsize': '13', 'family': 'Arial'}
sax.set_title('South Africa', **lab_props)
fax.set_title('Finland', **lab_props)

# Adjust limits
sax.set_ylim(-0.5, 1.0)
fax.set_ylim(-0.5, 1.0)

# Add y-label
fig.add_subplot(111, frameon=False)

ylab = "Pearson's correlation"
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid('off')

# Adjust the gaps between plots
plt.subplots_adjust(left = 0.08, right=0.98, wspace = 0.1, hspace = 0.2)
plt.ylabel(ylab, labelpad=10, **lab_props)

stat_cols = ['S_PCORR', 'I_PCORR', 'T_PCORR', 'F_PCORR']

# Drop NaN values
sa_clean = sa.dropna(subset=[stat_cols])
fi_clean = fi.dropna(subset=[stat_cols])
    
# Test for normality
# ------------------

# All stats
stat_cols = ['I_PCORR', 'T_PCORR', 'F_PCORR']
ins, ts, fs = sa_clean[stat_cols[0]], sa_clean[stat_cols[1]], sa_clean[stat_cols[2]]
inf, tf, ff = fi_clean[stat_cols[0]], fi_clean[stat_cols[1]], fi_clean[stat_cols[2]]

print("SOUTH AFRICA")
for platform in stat_cols:
    print(platform)
    
    # Perform Kolmogorov-Smirnoff test
    ks_stat, ks_pval = kstest(rvs=sa_clean[platform], cdf='norm', N=len(sa_clean))
    print("Kolmogorov-Smirnoff test\np-value: %s\nstat_value: %s" % (ks_pval, ks_stat))
    
    # Perform Anderson Darling test
    statistics, crit_vals, sig_level = anderson(x=sa_clean[platform], dist='norm')
    
    print(statistics, '\n', crit_vals, '\n', sig_level)
    
    txt = parseADnormalResults(stats=statistics, critical_values=crit_vals, signif_level=sig_level)
    print(txt)
    
# Test for homoscedasticity (the standard deviation in all groups are similar)
b_stat, b_pval = bartlett(sa[stat_cols[0]], sa[stat_cols[1]], sa[stat_cols[2]])
print("Homoscedasticity p-value: %s" % b_pval)

print("FINLAND")

for platform in stat_cols:
    print(platform)
    
    # Perform Kolmogorov-Smirnoff test
    ks_stat, ks_pval = kstest(rvs=fi_clean[platform], cdf='norm', N=len(sa_clean))
    print("Kolmogorov-Smirnoff test\np-value: %s\nstat_value: %s" % (ks_pval, ks_stat))
    
    # Perform Anderson Darling test
    statistics, crit_vals, sig_level = anderson(x=fi_clean[platform], dist='norm')
    
    print(statistics, '\n', crit_vals, '\n', sig_level)
    
    txt = parseADnormalResults(stats=statistics, critical_values=crit_vals, signif_level=sig_level)
    print(txt)
    
# Test for homoscedasticity (the standard deviation in all groups are similar)
b_stat, b_pval = bartlett(fi[stat_cols[0]], fi[stat_cols[1]], fi[stat_cols[2]])
print("Homoscedasticity p-value: %s\n" % b_pval)

# =========================================================================================================
# OUTCOME: The variances differ ==> Need to use Kruskal-Wallis test which is a non-parametric test of ANOVA
# =========================================================================================================

# South Africa
print("South Africa - Kruskal-Wallis")
kw_stat, kw_pval = kruskal(ins, ts, fs, nan_policy='omit')
print("Difference between all groups\nKruskal-Wallis p-value: %s\tstatistic: %s\n" % (kw_pval, kw_stat))

# Finland
print("Finland - Kruskal-Wallis")
kw_stat, kw_pval = kruskal(inf, tf, ff, nan_policy='omit')
print("Difference between all groups\nKruskal-Wallis p-value: %s\tstatistic: %s\n" % (kw_pval, kw_stat))

# Make Kruskall-Wallis with Dunn's test to compare the groups
# -----------------------------------------------------------

# Here we use Dunn's test to make a post-hoc test for differences. 
# The samples are independent, hence, we use sidak adjustment for p-values

# ===================
# R-TESTS
# ===================

def dunnTestR(groups=None, method='hs', kw=True):
    
    dunn = importr("dunn.test")
    # Runn dunn test
    results = dunn.dunn_test(x=groups, method='hs', kw=True)
    
    # Convert output to string and parse relevant information
    s = str(results)
    
    # Split lines
    split = s.split('\n\r')

    dunn_txt = "".join(split)
    return dunn_txt

# ------------
# SOUTH AFRICA
# ------------
print("\nSOUTH AFRICA\n")
sa_groups = [ins, ts, fs]

# Run dunn test in R
sa_dunn = dunnTestR(groups=sa_groups, method='hs')
print(sa_dunn)

# ------------
# FINLAND
# ------------

print("\nFINLAND\n")
sa_groups = [inf, tf, ff]

# Run dunn test in R
sa_dunn = dunnTestR(groups=sa_groups, method='hs')
print(sa_dunn)


