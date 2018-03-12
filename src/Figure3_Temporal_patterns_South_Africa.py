"""
Script for producing Figure 3. 
Temporal patterns and correlations between official visitor statistics and social media user-days in South African national parks.

Code associated to following article:
    
    Henrikki Tenkanen, Enrico Di Minin, Vuokko Heikinheimo, Anna Hausmann, Marna Herbst, Liisa Kajala & Tuuli Toivonen. (2017).
    Instagram, Flickr, or Twitter: Assessing the usability of social media data for visitor monitoring in protected areas. 
    Scientific Reports 7, 17615. doi:10.1038/s41598-017-18007-4

Author: 
    Henrikki Tenkanen, Digital Geography Lab, Department of Geosciences and Geography, University of Helsinki.

Requirements:
    geopandas
    pandas
    numpy
    datetime
    geoplot
    pandas
    matplotlib
    scipy
    
Created on:
    Sun May 21 17:00:06 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/

"""

import geopandas as gpd
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import os
from datetime import datetime
import some_utils as su
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import pearsonr, spearmanr

def createDatetimeIndex(df, time_col):
    """
    Creates a Pandas DatetimeIndex based on a column having time information.
    
    Parameters
    ----------
    
    df : pd.DataFrame
        Pandas DataFrame containing the data.
    time_col : str
        Name of the column containing the time information.
        
    Returns
    -------
    pd.DataFrame
    
    """
    df = df.sort_values(by=time_col)
    df = df.reset_index(drop=True)
    df['time'] = pd.to_datetime(df[time_col])
    df = df.set_index(pd.DatetimeIndex(df['time']))
    return df

def parseTimeCharacteristics(df, time_col='index', year_col='year', month_col='month', week_col='week', day_col='day', date_col='date'):
    """
    Parses various temporal characteristics based on Pandas DateTime index.
    
    Parameters
    ----------
    
    df : pd.DataFrame
        DataFrame containing the data where characteristics are parsed.
    time_col : str
        (Optional) Name of the column containing the pd.DateTime data. If 'index', use pd.DatetimeIndex for parsing the values. 
    year_col | month_col | week_col | day_col: str
        (Optional) Name for the column that will be created into the DataFrame.
    date_col : str
        Name for the column having a human readable date in the format of %Y-%m-%d.
        
    Returns
    -------
    pd.DataFrame
    """
    if time_col=='index':
        df[year_col] = df.index.year
        df[month_col] = df.index.month
        df[week_col] = df.index.week
        df[day_col] = df.index.day
        df[date_col] = df.index.strftime("%Y-%m-%d")
    elif time_col:
        df[year_col] = df[time_col].year
        df[month_col] = df[time_col].month
        df[week_col] = df[time_col].week
        df[day_col] = df[time_col].day
        df[date_col] = df[time_col].strftime("%Y-%m-%d")
    else:
        raise "Specify a column name having the time information."
    return df

def calculateSomeTemporalStats(df, t_grouping_col, area_grouping_col=None, userid_col='userid', date_col='date', likes_col='likes', post_cnt_col='post_cnt', user_cnt_col='user_cnt', user_days_col='user_days'):
    """
    Calculates temporal statistics about social media data. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the data
    t_grouping_col : str
        Name of the column that will be used to group the data based on time (e.g. day, month, week)
    area_grouping_col : str
        (Optional) Name of the column that will be used to group the data based on specified area.
    userid_col: str
        Column having unique identifier for a user.
    date_col: str
        Column having date in format such as "2015-12-24" ==> Parse from time column
    likes_col: str
        Column having information about number of likes of the post. 
    """
    
    # Calculate statistics
    stats = pd.DataFrame()

    # Group by t_grouping_col (and area_grouping_col if found)
    if area_grouping_col:
        grouped = df.groupby(by=[t_grouping_col, area_grouping_col])
    else:
        grouped = df.groupby(t_grouping_col)

    # Iterate over df and calculate stats
    for k, rows in grouped:
        # Calculate number of posts
        post_cnt = len(rows)
        # Calculate number of visitors
        user_cnt = len(rows[userid_col].unique())
        # Daily users 
        dgrouped = rows.groupby(date_col)
        dusersum = 0
        for date, drows in dgrouped:
            dusers = len(drows[userid_col].unique())
            dusersum += dusers
        # Append to statistics df
        if isinstance(k, list) or isinstance(k, tuple):
            stats = stats.append([list(k) + [post_cnt, user_cnt, dusersum]]) 
        else:
            stats = stats.append([[k, post_cnt, user_cnt, dusersum]])
        
    if isinstance(k, list) or isinstance(k, tuple):
        stats.columns = [t_grouping_col, area_grouping_col, post_cnt_col, user_cnt_col, user_days_col]
    else:
        stats.columns = [t_grouping_col, post_cnt_col, user_cnt_col, user_days_col]
    
    # Return statistics
    return stats

def fixTimeContinuity(df, id_col=None, day_col=None, month_col=None, year_col=None, start_day=None, end_day=None, start_month=None, end_month=None, start_year=None, end_year=None):
    """Fixes time data in a way that there are no gaps in the temporal information. E.g. if the data misses altogether data from January and April, the function will add 0 values to those months."""
    
    # Df for fixed data
    fixed = pd.DataFrame()
    if month_col:
        months = [m for m in range(start_month, end_month+1)]
        if min(months)< 1:
            raise "Month cannot be less than 1, or what do you think? Check your 'start_month' parameter!"
        elif max(months)>12:
            raise "There is only 12 months in a year. Check your 'end_month' parameter!"
        # Create DataFrame for continuous Months
        months = pd.DataFrame(months, columns=[month_col])
        
        # If there is data from multiple parks 
        if id_col:
            # Group by identifier col
            grouped = df.groupby(id_col)
            # Iterate over groups and fill the time gaps
            for k, rows in grouped:
                fix = rows.merge(months, left_on=month_col, right_on=month_col, how='outer')
                # Fill id_col with id_col values
                id_col_val = rows[id_col].head(1).values[0]
                fix[id_col] = fix[id_col].fillna(value=id_col_val)
                # Fill NaN with 0
                fix = fix.fillna(value=0)
                # Sort by month_col
                fix = fix.sort_values(by=month_col)
                # Add data to fixed DataFrame
                fixed = fixed.append(fix)
        else:
            fix = df.merge(months, left_on=month_col, right_on=month_col, how='outer')
            # Fill NaN with 0
            fix = fix.fillna(value=0)
            # Sort by month_col
            fix = fix.sort_values(by=month_col)
            # Add data to fixed DataFrame
            fixed = fixed.append(fix)
            
    # Reset index
    fixed = fixed.reset_index(drop=True)
    return fixed

def normalizeColumns(df, src_cols, target_cols, sum_column=None):
    """
    Helper function to normalize the values of a column to scale 0.0-1.0. 
    If 'sum_column' is passed, the values will be normalized according that column. 
    """
    df = df.copy()
    # Calculate Percentage of the source column based on 
    for src, target in zip(src_cols, target_cols):
        if not sum_column:
            # Calculate the total sum of a column
            tot_sum = df[src].sum()
        else:
            tot_sum = df[sum_column].sum()
        # Normalize the column by total sum
        df.loc[df.index, target] = df[src] / tot_sum
    return df

def plotSomeCurves(df, x_cols, y_cols, fig=None, ax=None, props=None):
    """
    Helper function to plot the temporal patterns of official national park visitor statistics and social media user-days from different platforms.
    """
    
    # Create an empty canvas and axis for Figure
    if ax == None and fig == None:
        fig, ax = plt.subplots()
    
    # Prepare the columns to be plotted
    # ---------------------------------
    x_cols, y_cols = getColumns(x_cols, y_cols)
    
    # Plot curves
    for idx, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
        #print("Plotting:", x_col, y_col)
       
        # If properties does not exist for the plot ==> Plot with defaults
        if not props or len(props) == 0:
            df.plot(x=x_col, y=y_col, ax=ax)
        else:
            # If they do apply them
            
            # Get properties
            kind, style, dashes, alpha, lw, color, legend = getProperties(props, idx=idx)
            
            # Plot it! 
            if dashes:
                p = df.plot(x=x_col, y=y_col, kind=kind, style=style, dashes=dashes, alpha=alpha, lw=lw, c=color, legend=legend, ax=ax)
            else:
                p = df.plot(x=x_col, y=y_col, kind=kind, style=style, alpha=alpha, lw=lw, c=color, legend=legend, ax=ax)
    return fig, ax, p

def getColumns(x_cols, y_cols):
    """Helper function to parse equal number of x and y columns in lists."""
    # Check if x_cols is the same for all y_cols
    if isinstance(x_cols, str):
        # Check if y_cols is a list
        if isinstance(y_cols, list):
            x_cols = [x_cols for i in range(len(y_cols))]
        # If not plot only single plot with lists of a single value
        else:
            x_cols = [x_cols]
    
    # Check if y_cols is the same for all x_cols
    if isinstance(y_cols, str):
        # Check if y_cols is a list
        if isinstance(x_cols, list):
            y_cols = [y_cols for y_cols in range(len(x_cols))]
        # If not plot only single plot with lists of a single value
        else:
            y_cols = [y_cols]
    return x_cols, y_cols

def getProperties(props, idx):
    """Helper function to parse visual properties from props dictionary"""
    
    # Get the properties for the plot
    if len(props) > 1:
        p = props[idx]
        if 'kind' in p:
            kind = p['kind']
        else:
            kind = 'line'
        if 'lw' in p:
            lw = p['lw']
        else:
            lw = 1
        if 'style' in p:
            style = p['style']
        else:
            style = '-'
        if 'color' in p:
            color = p['color']
        else:
            color = 'b'
        if 'dashes' in p:
            dashes = p['dashes']
        else:
            dashes = None
        if 'alpha' in p:
            alpha = p['alpha']
        else:
            alpha = 1.0
        if 'legend' in p:
            legend=p['legend']
        else:
            legend=True
        return kind, style, dashes, alpha, lw, color, legend
    
def calculateCorrelations(df, col1, col2):
    """Helper functino to calculate Pearson and Spearman correlations"""
    # Calculate Pearson correlation
    pcorrelation, pp_value = pearsonr(df[col1], df[col2])
    # Calculate Spearman correlation
    scorrelation, sp_value = spearmanr(df[col1], df[col2])
    return pcorrelation, pp_value, scorrelation, sp_value

def plotPropMonthlyAppearance(ax, month_values, xllimit, xulimit, xstep, yllimit, yulimit, ystep, legend, legend_loc, legend_size, label_size=10, grid_linewidth=0.25):
    """Helper function to adjust the visual appearance of the plot on temporal patterns."""
    
    # Month tick labels
    ax.xaxis.set_ticks(np.arange(1,13,1))
    m = list(month_values)
    ax.set_xticklabels(m) 
    ax.tick_params(axis='both', direction="out",labelsize=label_size, pad=2.5)
    
    # Fix and standardize y-scale
    ax.yaxis.set_ticks(np.arange(yllimit,yulimit, ystep))
    
    # Fix legend size
    if legend:
        plt.legend(loc=legend_loc,prop={'size':legend_size})

    ax = adjustGridlines(ax=ax)
    return ax

def adjustGridlines(ax, grid_linewidth=0.5):
    """Helper function to adjust the aesthetics of the gridlines"""
    gridlines = ax.get_xgridlines()
    gridlines.extend( ax.get_ygridlines() )

    for line in gridlines:
                line.set_linewidth(grid_linewidth)
                line.set_linestyle('dotted')
    ax.grid(True)
    return ax

def autocorrelationPlot(data, columns=['Official', 'Instagram', 'Twitter', 'Flickr']):
    """Helper function to create autocorrelation plots for specified columns"""
    for col in columns:
        autocorrelation_plot(data[col])


# Paremeters
# ----------

# Use Seaborn White grid style
plt.style.use('seaborn-whitegrid')

sa_name_convert = {'Augrabies Falls': 'Augrabies',
                'Golden Gate Highlands': 'Golden Gate',
                'Addo Elephant' : 'Addo',
                'Kalahari Gemsbok': 'Kgalagadi'
                }


# Calculate Pearson's r
calc_pearson = True

# Plot autocorrelations
autocorr_plot = False
autocorr_cols = ['Official', 'Instagram']
    
# Options for Pandas - Show 40 columns
pd.set_option('display.max_columns', 40)

# Select data for year 2014
start, end = datetime(2014,1,1,0,0,0), datetime(2014,12,12,23,59,59)

# Data folder
fold = "/home/social-media/SA"

# Filepaths
flickrfp = os.path.join(fold, "Flickr_SANParksAll_2008-2016_Feb.shp")
instafp = os.path.join(fold, "Instagram_SouthAfrican_NPs_2013-2015_October.shp")
twitterfp = os.path.join(fold, "DOLLY_SA_NPs_tweets_TEMPORAL-ONLY_2012-2016_ALL_WithinPark_w_PARKNAMES_CLEANED_without_Noordhoek.shp")
statsfp = "SANParks_monthly_visitors_national_international_2014.csv"
outputdir = "/home/results/SA"

# Read files
stats = pd.read_csv(statsfp, sep=';', encoding='latin1')
insta = gpd.read_file(instafp)
flickr = gpd.read_file(flickrfp)
twitter = gpd.read_file(twitterfp)

# Rename 'Bontekok' to 'Bontebok'
stats = stats.replace({'Bontekok': 'Bontebok'})

# There seems to be some duplicate records (remove those)
stats = stats.drop_duplicates()

# Rename Twitter columns
twitter = twitter.rename(columns={'id': 'photoid', 'u_id': 'userid'})

# Create 
twitter['Distance'] = 0

# Select relevant columns from social media datasets
cols = ['NAME', 'time_local', 'userid', 'photoid', 'geometry', 'Distance']
insta = insta[cols]
flickr = flickr[cols]
twitter = twitter[cols]

# Create datetime indexes based on local time 
insta = createDatetimeIndex(insta, time_col='time_local')
flickr = createDatetimeIndex(flickr, time_col='time_local')
twitter = createDatetimeIndex(twitter, time_col='time_local')
stats = createDatetimeIndex(stats, time_col='Date')

# Parse temporal characteristics
insta = parseTimeCharacteristics(df=insta)
flickr = parseTimeCharacteristics(df=flickr)
twitter = parseTimeCharacteristics(df=twitter)

# Select data based on time window
insta = insta[start:end]
flickr = flickr[start:end]
twitter = twitter[start:end]

# Rename columns
insta['NAME'] = insta['NAME'].replace(to_replace=sa_name_convert)
twitter['NAME'] = twitter['NAME'].replace(to_replace=sa_name_convert)
flickr['NAME'] = flickr['NAME'].replace(to_replace=sa_name_convert)


# Select data that is within ~1 km buffer from the park boundaries 
# Label name for the buffer that will be used in the output filenames
buf_label = "within"

# Buffer in Decimal degrees
# -------------------------
buffer_dd = 0.01  
distance_column = 'Distance'

insta = insta.ix[insta[distance_column] < buffer_dd]
flickr = flickr.ix[flickr[distance_column] < buffer_dd]
twitter = twitter.ix[twitter[distance_column] < buffer_dd]

# Calculate monthly statistics
month_col = 'month'
park_name_col = 'NAME'

instam = calculateSomeTemporalStats(df=insta, t_grouping_col=month_col, area_grouping_col=park_name_col)
flickrm = calculateSomeTemporalStats(df=flickr, t_grouping_col=month_col, area_grouping_col=park_name_col)
twitterm = calculateSomeTemporalStats(df=twitter, t_grouping_col=month_col, area_grouping_col=park_name_col)

# Ensure that each month has values (if no social media content was published, add zero values)
instam = fixTimeContinuity(instam, id_col=park_name_col, month_col=month_col, start_month=1, end_month=12)
flickrm = fixTimeContinuity(flickrm, id_col=park_name_col, month_col=month_col, start_month=1, end_month=12)
twitterm = fixTimeContinuity(twitterm, id_col=park_name_col, month_col=month_col, start_month=1, end_month=12)

# Rename columns for insta and flickr
instam.columns = list(instam.columns[0:2])+['i'+ col for col in instam.columns[2:]]
flickrm.columns = list(flickrm.columns[0:2])+['f'+ col for col in flickrm.columns[2:]]
twitterm.columns = list(twitterm.columns[0:2])+['t'+ col for col in twitterm.columns[2:]]

# Join datasets based on 'NAME' and 'Month'
join = instam.merge(flickrm, on=[park_name_col, month_col], how='outer')
join = join.merge(twitterm, on=[park_name_col, month_col], how='outer')
join = join.merge(stats, on=[park_name_col, month_col], how='outer')

# Fill NaN with 0.0
join = join.fillna(0.0)

# Order by 'NAME' and 'Month'
join = join.sort_values(by=[park_name_col, month_col])

# Calculate TOTAL Social media user-days 
join['suser_days'] = join['iuser_days'] + join['fuser_days'] + join['tuser_days']

# Make DataFrame for statistics
c = pd.DataFrame()

# Remove parks without official statistics
removable_parks = ['Knysna', 'Groenkloof']
join = join.ix[~join[park_name_col].isin(removable_parks)]

# Calculate the total visitors per park
tot_vis = 'TotOfficialVisitors'
idx_val = 'idx'
join[tot_vis] = None
join[idx_val] = None
visitor_col = 'Total'
grouped = join.groupby(park_name_col)
i = 1
for idx, rows in grouped:
    # Total sum
    vis_sum = rows[visitor_col].sum()
    join.loc[rows.index, tot_vis] = vis_sum
    join.loc[rows.index, idx_val] = i
    i+=1

# Sort DataFrame based on number of official visitors (descending(tot_vis))
join = join.sort_values(by=[tot_vis, month_col], ascending=[False, True]).copy()

# Select parks
sel_parks = list(join[park_name_col].unique())

# Plot the temporal patterns of all South African national parks as subplots
# ==========================================================================

fig, axarray = plt.subplots(nrows=4, ncols=6, figsize=(15.5,9.0), sharey=True)

row_idx = 0
col_idx = 0
# Iterate over parks and plot information
for i, park in enumerate(sel_parks):
    rows = join.ix[join[park_name_col]==park]
    print(park)
    # Get ax 
    ax = axarray[row_idx][col_idx]
    
    # Name for all platforms combined (some)
    sname = "All platforms"

    # Normalize the official statistics
    # ---------------------------------
    rows = normalizeColumns(df=rows, src_cols=['Total'], target_cols=['Official'])
        
    # Normalize the social media user-days based on total user-days (sum) of all social media platforms
    # -------------------------------------------------------------------------------------------------
    source_cols = ['iuser_days', 'fuser_days', 'tuser_days', 'suser_days']
    
    # Names for user-day-share output columns (should be in the same order as the list above)
    target_cols = ['Instagram', 'Flickr', 'Twitter', sname]
    
    # Calculate normalized user days for each platform (use 'suser_days' column for normalization for each column)
    rows = normalizeColumns(df=rows, src_cols=source_cols, target_cols=target_cols, sum_column='suser_days')
    
    # PLOT normal Temporal Curves
    # ---------------------------

    # X-columns to be plotted 
    # If contains a string plotting assumes that x column is the same for every y-column
    # If contains a list plotting assumes that x column is defined separately for each y-column (lenght of x and y columns need to match, of course)
    x_cols = month_col

    # Y-columns to be plotted    
    # If contains a string plotting assumes that y column is the same for every x-column
    # If contains a list plotting assumes that y column is defined separately for each x-column (lenght of y and x columns need to match, of course)
    y_cols = ['Official', sname, 'Instagram', 'Twitter', 'Flickr']
    
    # Properties for each plot 
    # They should be in the same order as the plots in the x_cols and y_cols and have:
        # - a single property if props are the same for all plots
        # - have as many properties as there are plots if you want to apply different properties for plots
    props = {0 : {'kind': 'line', 'style': '-', 'lw': 1.25, 'color': 'red', 'legend': False},
             1 : {'kind': 'line', 'style': '--', 'dashes': (5, 1), 'lw': 1.25, 'color': '#377eb8', 'legend': False},
             2 : {'kind': 'line', 'style': '--', 'dashes': (1, 1), 'lw': 0.8, 'alpha': 0.9, 'color': '#e41a1c', 'legend': False},
             3 : {'kind': 'line', 'style': '--', 'dashes': (1, 1), 'lw': 0.8, 'alpha': 0.9, 'color': 'green', 'legend': False},
             4 : {'kind': 'line', 'style': '--', 'dashes': (1, 1), 'lw': 0.8, 'alpha': 0.9, 'color': '#331900', 'legend': False},
             }
             
    # Plot the data
    # -------------
    fig, ax, p = plotSomeCurves(fig=fig, ax=ax, df=rows, x_cols=x_cols, y_cols=y_cols, props=props)
    
    # Add fill between Social media and official statistics
    ax.fill_between(rows[x_cols], rows[y_cols[0]], rows[y_cols[1]], alpha=0.3, facecolor='#377eb8')
    
    # Sequencial number
    seq_i = i+1
    
    # Calculate statistics
    # --------------------
        
    # Columns that will be used to calculate statistics
    col1, col2 = 'Official', sname
    # Pearson corr position
    ppos_x = 9.95; ppos_y = 0.41
    
    if calc_pearson:
        
        # Correlations
        # ............
        
        # Parse values for social media monthly total share values and official statistics monthly share values
        pcorrelation, pp_value, scorrelation, sp_value = calculateCorrelations(df=rows, col1=col1, col2=col2)
        
        # Round values and parse texts
        pcorrelationtxt = "%0.2f" % (np.round(pcorrelation, 2))
        lab_props = {'fontsize': '14', 'family': 'Arial', 'color':'#404040'}
        if pcorrelation < 0:
            ax.text(ppos_x-0.35, ppos_y, pcorrelationtxt, **lab_props)
        else:
            ax.text(ppos_x, ppos_y, pcorrelationtxt, **lab_props)
                
        # Pearson mark for first graph
        if i == 0:
            ax.text(ppos_x-1.25, ppos_y, "r =", **lab_props)
    
    # Get Month values as integers
    m_values = rows[month_col].astype(int).values
    
    # Adjust plot visual elements 
    yaxis_limit = 0.46
    ax = plotPropMonthlyAppearance(ax=ax, month_values=m_values, xllimit=1, xulimit=13, xstep=1, yllimit=0.0, yulimit=yaxis_limit, ystep=0.1, legend=False, legend_loc=2, legend_size=9, label_size=10, grid_linewidth=0.1)
    
    # Set ylimit
    ax.set_ylim(0, yaxis_limit)
    
    # Exclude labels at this point
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Add park name
    lab_props = {'fontsize': '14', 'family': 'Arial'}
    llabel_x = 1.24
    ax.text(llabel_x, yaxis_limit-0.05, park, **lab_props)
    
    # Add sequencial number
    if i == 0:
        ax.text(llabel_x, ppos_y-0.055, "ID: %s" % seq_i, **{'fontsize': '11.5', 'family': 'Arial', 'color':'gray'})
    else:
        ax.text(llabel_x, ppos_y-0.055, "%s" % seq_i, **{'fontsize': '11.5', 'family': 'Arial', 'color':'gray'})
        
    # Column index is not used here
    col_idx += 1
    
    if col_idx==6:
        # Update indices
        row_idx+=1
        col_idx=0
    
    # Save the legend in the last round ('Agulhas' is the last park)
    if park == 'Agulhas':
        lgnd = ax
        
    # Plot the autocorrelations if wanted (only to the screen)
    if autocorr_plot:
        autocorrelationPlot(data=rows, columns=autocorr_cols)

# ================================================
# Adjust legend and general appearance of the plot
# ================================================
        
# Use the third last axis for legend
ax = axarray[-1][-3]
ax.grid('off')
ax.axis('off')
lgnd.legend(bbox_to_anchor=(1.1, 1.0), labelspacing=0.7, prop={'size':14, 'family': 'Arial'})

# Add rectange to legend about the filling
rect = patches.Rectangle((0,0),1,1, facecolor='#377eb8', alpha=0.3)

lgnd_font_s = 14
dlgnd_y = 1.0
anchor = (2.85, dlgnd_y)
ax.legend(handles=[rect], labels=['Official vs All platforms (difference)'], prop={'size':lgnd_font_s, 'family': 'Arial'}, bbox_to_anchor=anchor)

# ================
# Add legend for Pearson's correlation coefficient
plgnd_y = dlgnd_y-0.72
plgnd_x = 1.18
ax.text(plgnd_x, plgnd_y, "r", **{'size':14, 'family': 'Arial'})
ax.text(plgnd_x + 0.19, plgnd_y, "Pearson's correlation", **{'size':14, 'family': 'Arial'})

# Add legend for Park ID
ax.text(plgnd_x -0.02, plgnd_y-0.09, "ID", **{'size':14, 'family': 'Arial', 'color': 'gray'})   
ax.text(plgnd_x + 0.19, plgnd_y-0.09, "Park id-number", **{'size':14, 'family': 'Arial'})

# ====================

# Make the last two axes blank white
ax, ax1 = axarray[-1][-1], axarray[-1][-2]
ax.grid('off'); ax.axis('off')
ax1.grid('off'); ax1.axis('off')

# Add common x and y-label for whole image 
# ========================================

# Add big axes, hide frame
fig.add_subplot(111, frameon=False)

xlab = 'Month'
ylab = 'Proportion of visitors'
lab_props = {'fontsize': '20', 'family': 'Arial'}
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid('off')

# Adjust the gaps between plots
plt.subplots_adjust(left = 0.05, right=0.95, bottom=0.10, top=0.95, wspace = 0.1, hspace = 0.2)
plt.xlabel(xlab, labelpad=12, **lab_props)
plt.ylabel(ylab, labelpad=12, **lab_props)

# Save the output figure
# ======================
# Output name
outfp = os.path.join(outputdir, 'SANParks_monthly_visitation_comparisons_selected_parks_2014.png')
plt.savefig(outfp, dpi=600)
