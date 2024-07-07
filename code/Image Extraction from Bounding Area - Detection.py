#!/usr/bin/env python
# coding: utf-8

# ### Library Pemetaan dan Operating System

# In[1]:


import os
from pathlib import Path
from PIL import Image
import shutil
import pyproj

import utility


# ### Library Plotting dan Analisis Numerik

# In[2]:


import pandas as pd
import numpy as np
import time
import random
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import folium
import branca.colormap as cm

from shapely.geometry import Polygon
from shapely.geometry import Point


# ### Library Machine Learning & Deep Learning

# In[3]:


from keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image


# In[ ]:


# padang_shape_path = r'Mapping/boundary/padangboundary.txt'
# with open(padang_shape_path, 'r') as file:
#     # Read all the lines of the file into a list
#     lines = file.readlines()

# padang_shape_list = []
# # Print the list of lines
# for cnt, line in enumerate(lines):
#     if(cnt > 0):
#         grid = False
#         line = line.replace('\n','').replace(',','.')
#         lat, lon = line.split(' ')[1], line.split(' ')[0]
#         float_lat, float_lon = float(lat), float(lon)
#         if(((float_lat * 1000000) % 5 == 0) and ((float_lon * 1000000) % 5 == 0)):
#             grid = True
# #             print(float_lat, float_lon)
#         if(grid == False):
#             padang_shape_list.append([float_lat, float_lon])


# In[9]:


# Pendefinisian Variabel Awal

# 1. Bounding Box Area yang akan dianalisa

pamulang_poly = Polygon([[-6.323284349631147, 106.70161466324525], [-6.323435986682789, 106.72861871041171], 
                   [-6.3050875808771965, 106.73105975422338], [-6.304935938452187, 106.74494319090218], 
                   [-6.339812526277561, 106.74799449566676], [-6.340419054802105, 106.75653814900755],
                  [-6.359979216832972, 106.76996388997166], [-6.360282469305901, 106.7177865784975],
                  [-6.3385994670884305, 106.6991736194336], [-6.326013809755609, 106.70359801134224]])
jakarta_poly = Polygon([[-6.095916216660592, 106.68530340448586], [-6.102061047272174, 106.80477972048222], 
                       [-6.119812384154859, 106.80958623894185], [-6.113654968705785, 106.8580325812851],
                       [-6.09521792837518, 106.87657476117133], [-6.092529140348378, 106.96735418353099],
                       [-6.194516408555102, 106.96868377796712], [-6.251776710484931, 106.94406036759005],
                       [-6.25654313700135, 106.90364567039597], [-6.363435792037267, 106.90981062420524],
                       [-6.339608132221271, 106.84816108611255], [-6.364116566099691, 106.79473148643225],
                       [-6.226582020915292, 106.71527208177946], [-6.099228304233594, 106.68787228707161]])
padang_poly_rough = Polygon([[-0.8169522943332843, 100.29115393394389],[-0.8071627055897717, 100.2903794814684],
                            [-0.7906079369224775, 100.31129280041517],[-0.793221852188435, 100.3208780715991],
                            [-0.8193609131427843, 100.33590951959208],[-0.8126083385155769, 100.33743444909862],
                            [-0.8263313008188163, 100.35094096758509],[-0.8167470146320907, 100.3622690153479],
                            [-0.8792058239140979, 100.41803970354398],[-0.8637185667761207, 100.43081816833393],
                            [-0.9190852126116854, 100.4222991918073],[-0.9039853012385667, 100.4559878717081],
                            [-0.9152134472615293, 100.47225137465067],[-0.9341850608971407, 100.4648940767413],
                            [-0.9365081086306252, 100.49161268493849],[-0.9574154684897493, 100.49006378011546],
                            [-1.0046501413142748, 100.3642152630547],[-0.9059211921150548, 100.34175613984918], 
                            [-0.8493927980701989, 100.32200760192039],[-0.8203539150072587, 100.29064227925412], 
                            ])
padang_city_poly_rough = Polygon([[-0.9642021332463775, 100.35129330887598],[-0.9610346110266733, 100.36681819583903],
                                  [-0.9592920789508105, 100.37836409237548],[-0.9774826186930436, 100.38329509849804],
                                  [-0.9768268575952576, 100.39080100999477],[-0.9573725577234504, 100.39233134146762],
                                  [-0.9291890232625566, 100.37053202735413],[-0.9309495074604052, 100.35005169234938],
                                  [-0.9575532632415699, 100.35281420609273],[-0.9637317017819592, 100.35157834468123]
                                 ])

# 7. Directory tempat saving gambar dan csv
loc_name = 'PREFIX_NAMA GAMBAR HASIL GSV MINING'
output_dirs = 'OUTPUT DIRECTORY'
folium_map_center_lat = -0.907807699324931 # Default posisi di tengah Kota Padang
folium_map_center_lon = 100.37461998026899 # Default posisi di tengah Kota Padang


# ### Eksekusi GSV API Mining

# In[ ]:


foto_bangunan = utility.sv_mining(padang_city_poly_rough, loc_name , num_query = 1200, min_threshold = 0.65, 
          dirs = output_dirs,radius = 50, size = "256x256")


# In[ ]:


# Extracting data dari list foto_bangunan menjadi csv yang siap digunakan pada database

df = pd.DataFrame(foto_bangunan, columns = ['Index', 'Path', 'Lintang', 'Bujur', 'Heading', 'Tipologi'])
df.to_csv(f'{output_dirs}/{loc_name} 060724.csv', index = False)


# ### Calculate distances between epicentrum and each Building Node
# ### Then, calculating hypocentrum with pythagoras formula
# #### First, we define the earthquake scenarios

# In[10]:


# Earthquake Scenario (Padang 2009 EQ)
eq_depth = 90 # in km. DEFAULT GEMPA PADANG 30 SEPTEMBER 2009
eq_mw = 8.1 # magnitude from EQ history in Sumatra subduction zone
eq_lat, eq_lon = -0.7071, 99.9678 # DEFAULT KOORDINAT EPISENTRUM GEMPA PADANG 30 SEPTEMBER 2009


# In[ ]:


df = pd.read_csv(f'{output_dirs}/{loc_name} 050724.csv')


# In[ ]:


# Initiate new empty columns for filling
df["Epicentrum Dist (km)"], df["Hypocentrum Dist (km)"], df["PGA Bedrock (g)"], df["Surface PGA (gal)"], df["MMI"] = 0.0, 0.0, 0.0, 0.0, 0
df["Vulnerability_Class"], df["Vulnerability_MMI"] = "",""
df["D0"], df["D1"], df["D2"], df["D3"], df["D4"], df["D5"] = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
df["total"], df["max_prob"], df["damage_maxprob"] = 0.0, 0.0, 0.0


# In[ ]:


for i, (building_lat, building_lon, building_typology) in enumerate(zip(df['Lintang'],df['Bujur'],df['Tipologi'])):
    az, dist = utility.calc_azimuth_distance(eq_lat, eq_lon, building_lat, building_lon)
    dist_km = dist/1000
    hypocentrum = (dist_km**2 + eq_depth**2)**0.5
    pga = utility.calc_PGA_Young(eq_mw, eq_depth, dist_km, hypocentrum)
    surface_pga = utility.calc_PGA_surface(pga)
    mmi = np.round(utility.calc_MMI(surface_pga))
    
    # Filling the content of the main analysis table
    df.loc[i, "Epicentrum Dist (km)"], df.loc[i, "Hypocentrum Dist (km)"] = dist_km, hypocentrum
    df.loc[i, "PGA Bedrock (g)"], df.loc[i, "Surface PGA (gal)"], df.loc[i, "MMI"] = pga, surface_pga, mmi
    df.loc[i, "Vulnerability_Class"] = utility.randomize_building_vulnerability_class(building_typology)
    df.loc[i, "Vulnerability_MMI"] = df.loc[i, "Vulnerability_Class"] + str(df.loc[i, "MMI"])
    df.loc[i, "D0"], df.loc[i, "D1"], df.loc[i, "D2"], df.loc[i, "D3"], df.loc[i, "D4"], df.loc[i, "D5"] = utility.calculate_dmg_prob(df.loc[i, "Vulnerability_MMI"])
    df.loc[i, "total"] = df.loc[i, "D0"] + df.loc[i, "D1"] + df.loc[i, "D2"] + df.loc[i, "D3"] + df.loc[i, "D4"] + df.loc[i, "D5"]
    df.loc[i, "max_prob"] = df.loc[i, ["D0", "D1", "D2", "D3", "D4", "D5"]].max()
    df.loc[i, "damage_maxprob"] = str(df.loc[i, ["D5", "D4", "D3", "D2", "D1", "D0"]].idxmax())


# In[ ]:


df.head()


# In[ ]:


df.to_excel(f'{output_dirs}/{loc_name} 050724 - Analyzed.xlsx', index = False)


# ### Mapping with Folium

# In[11]:


df = pd.read_excel(f'{output_dirs}/{loc_name} 050724 - Analyzed.xlsx')

colormap_D1D5 = cm.LinearColormap(colors=['green','yellow','orange','red'], index=[0, 0.166, 0.33, 0.5],vmin=0,vmax=0.5)
maps = folium.Map(location=[folium_map_center_lat, folium_map_center_lon],zoom_start=11, tiles = 'OpenStreetMap')


# In[12]:

# Creating big red circle for epicentrum location for reference
folium.Circle(
            location=(eq_lat, eq_lon),
            radius = 500,
            fill=True,
            color='red',
            fill_opacity=0.4,
            popup = f'Epicentrum'
        ).add_to(maps)

# Creating hundreds of circles for building location and its maximum probability of damage experienced during a specified earthquake scenario
for loc, max_prob, damage_maxprob in zip(zip(df['Lintang'], df['Bujur']), df['max_prob'], df['damage_maxprob']):
    print(loc, max_prob, damage_maxprob)
    if(damage_maxprob == 'D0'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='white',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    elif(damage_maxprob == 'D1'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='darkgreen',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    elif(damage_maxprob == 'D2'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='green',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    elif(damage_maxprob == 'D3'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='yellow',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    elif(damage_maxprob == 'D4'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='orange',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    elif(damage_maxprob == 'D5'):
        folium.Circle(
            location=loc,
            radius = 120*max_prob,
            fill=True,
            color='red',
            fill_opacity=0.7,
            popup = f'{damage_maxprob}, {max_prob*100}% probability'
        ).add_to(maps)
    
maps.save(f'{output_dirs}/{loc_name} Folium Map.html')




