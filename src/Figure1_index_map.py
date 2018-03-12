# -*- coding: utf-8 -*-
"""
Script for producing index map showing the locations of South Africa and Finland in Figure 1. 
Final edits for producing the Figure 1 is done in CorelDraw software. 

Code associated to following article:
    
    Henrikki Tenkanen, Enrico Di Minin, Vuokko Heikinheimo, Anna Hausmann, Marna Herbst, Liisa Kajala & Tuuli Toivonen. (2017).
    Instagram, Flickr, or Twitter: Assessing the usability of social media data for visitor monitoring in protected areas. 
    Scientific Reports 7, 17615. doi:10.1038/s41598-017-18007-4
    
Data:
    TM_WORLD_BORDERS-0.3.shp can be obtained from http://thematicmapping.org/downloads/world_borders.php under Creative Commons BY-SA licence (https://creativecommons.org/licenses/by-sa/3.0/)

Author: 
    Henrikki Tenkanen, Digital Geography Lab, Department of Geosciences and Geography, University of Helsinki.

Requirements:
    geopandas
    geoplot
    matplotlib

Created on:
    Mon May 22 17:49:21 2017

License:
    Creative Commons BY 4.0. See details from https://creativecommons.org/licenses/by/4.0/
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# Filepaths
# ---------

# Country borders
c_fp = "TM_WORLD_BORDERS-0.3.shp"
outfp = "SA_FI_Index_map.png"
world = gpd.read_file(c_fp)

# Select Europe and Africa continents
fe = world.ix[world['REGION'].isin([150, 2])]

# Select Finland and South Africa
safi = world.ix[world['FIPS'].isin(['FI', 'SF'])]

# Exclude Russia
fe = fe.ix[fe['NAME']!='Russia']

# Define projection
crs = gcrs.Orthographic(central_latitude=30, central_longitude=25.3)

# Initialize figure
fig, ax = plt.subplots(figsize=(5,8), subplot_kw={'projection': crs})

# Plot
poly_kwargs = {'linewidth': 0.5, 'facecolor': '#E0E0E0' ,'edgecolor': 'gray'}
ax = gplt.polyplot(fe, ax=ax, projection=crs, **poly_kwargs)

poly_kwargs = {'linewidth': 0.5, 'facecolor': '#D2691E' ,'edgecolor': '#D2691E'}
ax = gplt.polyplot(safi, ax=ax, projection=crs, **poly_kwargs)

# Set xlim
ax.set_xlim(-4922112.37202865363, 2896459.9490097885)
ax.set_ylim(-5858192.8002962265, 4233856.9617742747)

plt.savefig(outfp, dpi=600)