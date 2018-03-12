# -*- coding: utf-8 -*-
"""
Script for producing the park count plots in Figure 1. 
Plotting national parks in South Africa and Finland according their visitor numbers. 
Final edits for producing the Figure 1 is done in CorelDraw software. 

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
    Sat Apr 22 15:16:24 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
import matplotlib.pyplot as plt
import os


def convertParkNames(df1, df2, df1_col, df2_col):
    """
    Helper function to rename Finnish park names from style 'Repoveden kansallispuisto' to 'Repovesi'.
    Requires 2 dataframes where df1 contains naming style 1 and df2 contains naming style 2. 
    """
    # Convert names for Finland NPs
    for idx, i in df1.iterrows():
        ip = i[df1_col]
        for idxj, j in df2.iterrows():
            jp = j[df2_col]
            if ip[0:4] == jp[0:4]:
                print("'%s': '%s'," % (ip,jp))

   
def plotParkVisitorCnts(park_points, country_bounds, visitor_col, legend_values, legend_labels, outfp, crs, scale_col=None, limits=(1,19), cmap='Blues', ax=None, poly_kwargs=None, point_kwargs=None, legend_kwargs=None, extent=None, figsize=None):
    """
    A helper function to produce Point maps where the size of the point is adjusted according the number of visitors in the park.
    """
    # Plot boundaries
    ax = gplt.polyplot(country_bounds, projection=crs, ax=ax, **poly_kwargs)
    
    # Create a point plot
    gplt.pointplot(park_points, ax=ax, projection=crs, 
                   scale=scale_col, limits=limits, 
                   hue=visitor_col, cmap=cmap, extent=extent, figsize=figsize,
                   legend=True, legend_var='hue',
                   legend_values=legend_values,
                   legend_labels=legend_labels,
                   legend_kwargs=legend_kwargs,
                   **point_kwargs)

    return ax
    
def plotPolygon(df, ax, projection, **kwargs):
    """
    Polygon plot with Cartopy library 
    """
    # Get geometries
    geoms = list(df['geometry'].values)
    print(geoms)
    
    # Plot geometries
    ax.add_geometries(geoms=geoms, crs=projection, **kwargs)
    return ax
    
              
# Conversion dictionaries for some park names
# -------------------------------------------
fin_name_convert = {'Itäinen Suomenlahti': 'It. Suomenlahti',
                'Kauhaneva-Pohjankangas': 'Kauha.-Pohjank.'}

sa_name_convert = {'Golden Gate Highlands': 'Golden Gate',
                'Kalahari Gemsbok': 'Kgalagadi',
                'Addo Elephant': 'Addo',
                'Augrabies Falls': 'Augrabies'}

# Filepaths
# ---------

# FI - Finland - Data sources
fin_stat_fp = "Finland_Visitor_Statistics_and_UserDays_2014.csv"
fin_parks_fp = "National_Park_POints.shp"
fin_bounds_fp = "TilastokeksusWFS_maakunta1000k_2016.shp"

# SA - South Africa - Data sources
san_stat_fp = "SANParks_Visitor_Statistics_and_UserDays_2014.csv"
sa_parks_fp = "SA_National_Parks_POINTS.shp"
sa_bounds_fp = "south_africa_regions.shp"

# Output folder
outdir = "/home/results/study_area"

# Read files
# ----------

# FI
fin_stat = pd.read_csv(fin_stat_fp, sep=';', encoding='latin1')
fin_parks = gpd.read_file(fin_parks_fp)
fin_bounds = gpd.read_file(fin_bounds_fp)

# SA
sa_stat = pd.read_csv(san_stat_fp, sep=';')
sa_parks = gpd.read_file(sa_parks_fp)
sa_bounds = gpd.read_file(sa_bounds_fp)

# Re-proejct to EPSG 4326
# -----------------------

# FI
fin_parks['geometry'] = fin_parks['geometry'].to_crs(epsg=4326)
fin_bounds['geometry'] = fin_bounds['geometry'].to_crs(epsg=4326)

# SA
sa_parks['geometry'] = sa_parks['geometry'].to_crs(epsg=4326)
sa_bounds['geometry'] = sa_bounds['geometry'].to_crs(epsg=4326)


# Rename few parks 
fin_parks['Nimi'] = fin_parks['Nimi'].replace(to_replace=fin_name_convert)
sa_parks['NAME'] = sa_parks['NAME'].replace(to_replace=sa_name_convert)

# Join visitor numbers to park points
# -----------------------------------

# FI
fi_data = fin_parks.merge(fin_stat, left_on='Nimi', right_on='Park')

# SA
sa_data = sa_parks.merge(sa_stat, left_on='NAME', right_on='Park')

# Remove parks without official statistics
# -----------------------------------------

# Remove Groenkloof (no official stats available)
sa_data = sa_data.ix[sa_data['Park']!='Groenkloof']

# Remove Natural parks from Finland (excluded as they are not national parks)
removable_parks = ['Aulanko', 'Linnansaari', 'Perämeri', 'Punkaharju', 'Tammisaari', 'Teijo']
fi_data = fi_data.ix[~fi_data['Park'].isin(removable_parks)]

# Sort values on alphapetical order
fi_data = fi_data.sort_values(by='Park').copy()
sa_data = sa_data.sort_values(by='Park').copy()

# Reset index
fi_data = fi_data.reset_index(drop=True)
sa_data = sa_data.reset_index(drop=True)

# Specify Park-ID
fi_data['Park-ID'] = fi_data.index + 1 
sa_data['Park-ID'] = sa_data.index + 1 
       
# Scale column
sa_data['scale'] = 200
fi_data['scale'] = 200

# Plot park visitors
# ------------------

# Flag for subplots
subplots = False

# Flag for Park-IDs
show_park_ids = True

if subplots:
    # Initialize figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6), subplot_kw={'projection': gcrs.Mercator()})

# Aesthetics
poly_kwargs = {'linewidth': 0.04, 'facecolor': '#A9A9A9' ,'edgecolor': 'k', 'zorder': -1}
point_kwargs = {'linewidth': 0.5, 'edgecolor': 'k', 'alpha': 0.9, 'color': '#D2691E'} ##1E90FF'}

# Plot visitor count map for South Africa
# =======================================

# Parameters
central_longitude, central_latitude = 25.3, -29.3
visitor_col = None 
scale_col = "Off_visitors"
legend_values = [10*1000, 100*1000, 500*1000, 1000*1000, 2000*1000]
legend_labels = ['10 thousand', '100 thousand', '500 thousand', '1 million', '2 million']
outfp = os.path.join(outdir, "SA_NP_visitors.png")

# Adjust the points sizes
min_point_size = 1
max_point_size = 24
limits=(min_point_size,max_point_size)
extent = [15, 35, -36, -20]
figsize=(10,8)

# Legend properties
legend_kwargs = {'loc': 'upper left', 'frameon': False, 'prop': {'size':12}}

# Colormap
cmap = "Blues"

# Define CRS 
crs = gcrs.PlateCarree()

# Plot South Africa
ax1 = plotParkVisitorCnts(park_points=sa_data, country_bounds=sa_bounds, 
                          scale_col=scale_col, visitor_col=visitor_col, 
                          legend_values=legend_values, legend_labels=legend_labels, 
                          outfp=outfp, crs=crs, limits=limits, cmap=cmap, 
                          point_kwargs=point_kwargs, poly_kwargs=poly_kwargs, 
                          legend_kwargs=legend_kwargs, extent=extent)

# Save Figure
plt.savefig(outfp, dpi=400)


# Plot visitor count map for Finland
# ==================================

# Parameters
central_longitude, central_latitude= 27.0, 63.1
visitor_col = None #'Off_visitors'
scale_col = "Off_visitors"
legend_values = [10*1000, 50*1000, 100*1000, 200*1000, 300*1000] 
legend_labels = ['10 thousand', '50 thousand', '100 thousand', '200 thousand', '300 thousand']#, '1 million', '2 million']
outfp = os.path.join(outdir, "FIN_NP_visitors.png")

# Legend properties
legend_kwargs = {'bbox_to_anchor': (0.37, 0.75), 'frameon': False, 'prop': {'size':12}}

# Adjust the points sizes
min_point_size = 1
max_point_size = 24
limits=(min_point_size,max_point_size)

# Colormap
cmap = "Blues"

# Define CRS
crs = gcrs.UTM(zone=35, southern_hemisphere=False)

# Plot
ax2 = plotParkVisitorCnts(park_points=fi_data, country_bounds=fin_bounds, visitor_col=visitor_col, scale_col=scale_col, legend_values=legend_values, legend_labels=legend_labels, outfp=outfp, crs=crs, cmap=cmap, limits=limits, point_kwargs=point_kwargs, poly_kwargs=poly_kwargs, legend_kwargs=legend_kwargs)

# Specify ylimit
# --------------

# Adjust ylimits 
ax2.set_ylim((-339064.19413892319, 639133.71398695))

# Save figure
plt.savefig(outfp, dpi=400)