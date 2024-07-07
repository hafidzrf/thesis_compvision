#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path
from PIL import Image
import shutil

import pyproj
from shapely.geometry import Polygon
from shapely.geometry import Point

import pandas as pd
import numpy as np
import time
import random
import requests

from keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image


# In[2]:


# This function calculates azimuth and distances in meters, given the information of initial coordinates (lat1, lon1)
# and terminus coordinate (lat2, lon2)

def calc_azimuth_distance(lat1, long1, lat2, long2):
    fwd_azimuth,back_azimuth,distance = geodesic.inv(long1, lat1, long2, lat2)
    return fwd_azimuth, distance

geodesic = pyproj.Geod(ellps='WGS84')


# In[3]:


def calc_PGA_Young(mag, depth, epic_dist, hyp_dist):
    c1, c2 = 0, 0
    c3, c4, c5 = -2.552, 1.45, -0.1
    zt = 1
    ln_pga = 0.2418 + 1.414*mag + c1 + c2*(10-mag)**3 + c3*np.log(epic_dist + 1.7818*np.exp(0.544*mag)) + 0.00607*depth + 0.3846*zt
    pga = np.exp(ln_pga)
    return pga


# In[4]:


# Based on Table 10 of SNI 1726-2019 of Earthquake-Resisting Buildings
def calc_PGA_surface(pga):
    if(pga < 0.1):
        surface_pga = 2.4 * pga
    elif(pga > 0.6):
        surface_pga = 1.1 * pga
    else:
        xp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        yp = [2.4, 1.9, 1.6, 1.4, 1.2, 1.1]
        surface_pga = np.interp(pga, xp, yp) * pga
    return surface_pga


# In[5]:

g_to_gal = 981
def calc_MMI(surface_pga):
    s_pga_gal = surface_pga * g_to_gal
    mmi = 0.1417 + 3.2335*np.log10(s_pga_gal)
    return mmi


# In[6]:


def randomize_building_vulnerability_class(building_typology):
    if(building_typology == 'Confined Masonry'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.27, 0.64, 0.09])[0]
    elif(building_typology == 'RC Infilled Masonry'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.27, 0.64, 0.09])[0]
    elif(building_typology == 'Timber Structure'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.09, 0.64, 0.27])[0]
    elif(building_typology == 'Unconfined Masonry'):
        kelas =  np.random.choice(['A','B','C'], 1, p=[0.09, 0.82, 0.09])[0]
    return kelas


# In[7]:


# Generating Random Coordinate within a rectangular bounding box. Inputs a list of 2 coordinates
# Outputs a random coordinate within a imaginary rectangular bounding box with a random uniform distribution

def generate_random_coord_rectangle(coords):
    lat_rand = np.random.uniform(coords[0], coords[1], size=None)
    lon_rand = np.random.uniform(coords[2], coords[3], size=None)
    file_name = str(np.round(lat_rand,7)) + ' - ' + str(np.round(lon_rand,7))
    #cache = np.array((lat_rand,lon_rand))
    return lat_rand,lon_rand, str(lat_rand) + ',' + str(lon_rand), file_name


# In[8]:


# Generating Random Coordinate within a polygon bounding box. Inputs a list of n coordinates
# Outputs N random coordinate within a polygon bounding box with a random uniform distribution

def random_points_in_polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    latlon = [[x.x, x.y] for x in points]
    return latlon


# In[9]:


def preprocess_image(path, target_size = (256,256)):
    img = image.load_img(path, target_size = target_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch
    return img_preprocessed


# In[10]:


def predict_buildings(path, model):
    img_preprocessed = preprocess_image(path, target_size = (256,256))
    prediction = np.squeeze(model.predict(img_preprocessed))
    return prediction


# In[11]:


def predict_typologies(path, model):
    img_preprocessed = preprocess_image(path,target_size = (256,256))
    prediction = np.squeeze(model.predict(img_preprocessed))
    prediction = typology_dict[np.argmax(prediction)]
    print(prediction)
    return prediction


# In[12]:


def GSV_query(meta_base, base_url, API_KEY, location_coord, heading, radius = 50, size = "256x256"):
    meta_params = {'key': API_KEY,
                   'location': location_coord}

    pic_params = {'key': API_KEY,
                  'location': location_coord,
                  'heading' : heading,
                  'radius' : radius,
                  'size': size}

    meta_response = requests.get(meta_base, params=meta_params)
    response = requests.get(base_url,params = pic_params)
    return meta_response, response


# In[13]:


def predict_folder(folder):
    for image_path in os.listdir(folder):
        path = folder + image_path
        print(path)
        predict_typology = predict_typologies(path, model_typology_detector)
        foto_bangunan.append([image_path, predict_typology])


# In[14]:
# 2. API Key untuk GCP dan metabase Google Street View Static API
API_KEY = 'YOUR_API_KEY'
meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
base_url = 'https://maps.googleapis.com/maps/api/streetview?'

# 3. Heading List GSV API (sudut POV pandang GSV yang dipertimbangkan)
heading_list = ["0", "180", "90", "270"]

# 4. Model AI untuk bangunan dan tipologi
model_building_detector = models.load_model('MODEL_BUILDING_DETECTOR_PATH')
model_typology_detector = models.load_model(MODEL_TYPOLOGY_DETECTOR_PATH')

# 5. Dictionary tipologi bangunan yang dipertimbangkan
typology_dict = {0 : 'Confined Masonry', 1 : 'RC Infilled Masonry',
                2 : 'Timber Structure', 3 : 'Unconfined Masonry'}
typology_acr_dict = {0 : 'CM', 1 : 'RC', 2 : 'TB', 3 : 'UC'}

# 6. Variabel foto_bangunan untuk menyimpan seluruh hasil capture GSV API, koordinat, dan deteksi model
foto_bangunan = []

def sv_mining(poly, loc_name, num_query, min_threshold = 0.60,  dirs = None,
             radius = 50, size = "256x256"):
    not_found = 0
    dapat = 0
    heading_list = ["0", "180", "90", "270"]
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print("Directory created successfully!")
    coords = random_points_in_polygon(poly, num_query)
    for cnt, (lat, lon) in enumerate(coords):
        file_name = str(np.round(lat,7)) + ' - ' + str(np.round(lon,7))
        location_coord = str(lat) + ',' + str(lon)
        heading = random.choice(heading_list)
        meta_response, response = GSV_query(meta_base, base_url,
                                            API_KEY, location_coord, heading, radius, size)
        if(meta_response.json().get("status") == 'OK'):
            img_path = dirs + str(cnt) + ' ' + loc_name + ' ' + file_name + '_' + heading + '.jpg'
            if(response.ok == True):
                if((cnt+1) % 50 == 0):
                    print('Pencarian gambar ke - ' + str(cnt+1))
                with open(img_path, 'wb') as file:
                    file.write(response.content)
                response.close()
                prediction = predict_buildings(img_path, model_building_detector)
                if prediction < min_threshold:
                    os.remove(img_path)
                else:
                    predict_typology = predict_typologies(img_path, model_typology_detector)
                    foto_bangunan.append([cnt, img_path, lat, lon, heading, predict_typology])
                    dapat+= 1
        else:
            not_found+= 1
    print("Total gambar tidak mampu di-query : "+ str(not_found) + " gambar")
    return foto_bangunan


# In[15]:
def calculate_dmg_prob(class_MMI):
    none, nearly_none, few, many, most, nearly_all, alls = 0.0, 0.015, 0.09, 0.35, 0.745, 0.97, 1.00
    
    #Dictionary Damage Class A-E, MMI 8
    damage_A_8 = {
        '0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few
    }

    damage_B_8 = {
        '0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none
    }

    damage_C_8 = {
        '0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none
    }

    damage_D_8 = {
        '0' : many + 7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none
    }

    damage_E_8 = {
        '0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none
    }
    damage_F_8 = {
        '0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none
    }
    
    #Dictionary Damage Class A-E, MMI 9
    damage_A_9 = {
        '0' : none, '1' : none, '2' : 1/3*few, '3' : 3*few, '4' : many, '5' : many
    }

    damage_B_9 = {
        '0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few
    }

    damage_C_9 = {
        '0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none
    }

    damage_D_9 = {
        '0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none
    }

    damage_E_9 = {
        '0' : many + 7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none
    }
    damage_F_9 = {
        '0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none
    }
    
    #Dictionary Damage Class A-E, MMI 10
    damage_A_10 = {
        '0' : none, '1' : none, '2' : none, '3' : 5/6*few, '4' : 2*few, '5' : most
    }

    damage_B_10 = {
        '0' : none, '1' : none, '2' : 1/3*few, '3' : 2*few, '4' : many+few, '5' : many
    }

    damage_C_10 = {
        '0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few
    }

    damage_D_10 = {
        '0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none
    }

    damage_E_10 = {
        '0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none
    }
    damage_F_10 = {
        '0' : many+7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none
    }
    
    #Dictionary Damage Class A-E, MMI 11
    damage_A_11 = {
        '0' : none, '1' : none, '2' : none, '3' : none, '4' : 5/6*few, '5' : most+2*few
    }

    damage_B_11 = {
        '0' : none, '1' : none, '2' : none, '3' : nearly_none, '4' : 8/3*few, '5' : most
    }

    damage_C_11 = {
        '0' : none, '1' : none, '2' : none, '3' : 4/3*few, '4' : many+2*few, '5' : many
    }
    damage_D_11 = {
        '0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few
    }

    damage_E_11 = {
        '0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none
    }
    damage_F_11 = {
        '0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none
    }
    
    if(class_MMI == 'A8'):
        d0, d1, d2, d3, d4, d5 = damage_A_8["0"], damage_A_8["1"], damage_A_8["2"], damage_A_8["3"], damage_A_8["4"], damage_A_8["5"]
    elif(class_MMI == 'A9'):
        d0, d1, d2, d3, d4, d5 = damage_A_9["0"], damage_A_9["1"], damage_A_9["2"], damage_A_9["3"], damage_A_9["4"], damage_A_9["5"]
    elif(class_MMI == 'A10'):
        d0, d1, d2, d3, d4, d5 = damage_A_10["0"], damage_A_10["1"], damage_A_10["2"], damage_A_10["3"], damage_A_10["4"], damage_A_10["5"]
    elif(class_MMI == 'A11'):
        d0, d1, d2, d3, d4, d5 = damage_A_11["0"], damage_A_11["1"], damage_A_11["2"], damage_A_11["3"], damage_A_11["4"], damage_A_11["5"]
    elif(class_MMI == 'B8'):
        d0, d1, d2, d3, d4, d5 = damage_B_8["0"], damage_B_8["1"], damage_B_8["2"], damage_B_8["3"], damage_B_8["4"], damage_B_8["5"]
    elif(class_MMI == 'B9'):
        d0, d1, d2, d3, d4, d5 = damage_B_9["0"], damage_B_9["1"], damage_B_9["2"], damage_B_9["3"], damage_B_9["4"], damage_B_9["5"]
    elif(class_MMI == 'B10'):
        d0, d1, d2, d3, d4, d5 = damage_B_10["0"], damage_B_10["1"], damage_B_10["2"], damage_B_10["3"], damage_B_10["4"], damage_B_10["5"]
    elif(class_MMI == 'B11'):
        d0, d1, d2, d3, d4, d5 = damage_B_11["0"], damage_B_11["1"], damage_B_11["2"], damage_B_11["3"], damage_B_11["4"], damage_B_11["5"]
    elif(class_MMI == 'C8'):
        d0, d1, d2, d3, d4, d5 = damage_C_8["0"], damage_C_8["1"], damage_C_8["2"], damage_C_8["3"], damage_C_8["4"], damage_C_8["5"]
    elif(class_MMI == 'C9'):
        d0, d1, d2, d3, d4, d5 = damage_C_9["0"], damage_C_9["1"], damage_C_9["2"], damage_C_9["3"], damage_C_9["4"], damage_C_9["5"]
    elif(class_MMI == 'C10'):
        d0, d1, d2, d3, d4, d5 = damage_C_10["0"], damage_C_10["1"], damage_C_10["2"], damage_C_10["3"], damage_C_10["4"], damage_C_10["5"]
    elif(class_MMI == 'C11'):
        d0, d1, d2, d3, d4, d5 = damage_C_11["0"], damage_C_11["1"], damage_C_11["2"], damage_C_11["3"], damage_C_11["4"], damage_C_11["5"]
    elif(class_MMI == 'D8'):
        d0, d1, d2, d3, d4, d5 = damage_D_8["0"], damage_D_8["1"], damage_D_8["2"], damage_D_8["3"], damage_D_8["4"], damage_D_8["5"]
    elif(class_MMI == 'D9'):
        d0, d1, d2, d3, d4, d5 = damage_D_9["0"], damage_D_9["1"], damage_D_9["2"], damage_D_9["3"], damage_D_9["4"], damage_D_9["5"]
    elif(class_MMI == 'D10'):
        d0, d1, d2, d3, d4, d5 = damage_D_10["0"], damage_D_10["1"], damage_D_10["2"], damage_D_10["3"], damage_D_10["4"], damage_D_10["5"]
    elif(class_MMI == 'D11'):
        d0, d1, d2, d3, d4, d5 = damage_D_11["0"], damage_D_11["1"], damage_D_11["2"], damage_D_11["3"], damage_D_11["4"], damage_D_11["5"]
    elif(class_MMI == 'E8'):
        d0, d1, d2, d3, d4, d5 = damage_E_8["0"], damage_E_8["1"], damage_E_8["2"], damage_E_8["3"], damage_E_8["4"], damage_E_8["5"]
    elif(class_MMI == 'E9'):
        d0, d1, d2, d3, d4, d5 = damage_E_9["0"], damage_E_9["1"], damage_E_9["2"], damage_E_9["3"], damage_E_9["4"], damage_E_9["5"]
    elif(class_MMI == 'E10'):
        d0, d1, d2, d3, d4, d5 = damage_E_10["0"], damage_E_10["1"], damage_E_10["2"], damage_E_10["3"], damage_E_10["4"], damage_E_10["5"]
    elif(class_MMI == 'E11'):
        d0, d1, d2, d3, d4, d5 = damage_E_11["0"], damage_E_11["1"], damage_E_11["2"], damage_E_11["3"], damage_E_11["4"], damage_E_11["5"]
    elif(class_MMI == 'F8'):
        d0, d1, d2, d3, d4, d5 = damage_F_8["0"], damage_F_8["1"], damage_F_8["2"], damage_F_8["3"], damage_F_8["4"], damage_F_8["5"]
    elif(class_MMI == 'F9'):
        d0, d1, d2, d3, d4, d5 = damage_F_9["0"], damage_F_9["1"], damage_F_9["2"], damage_F_9["3"], damage_F_9["4"], damage_F_9["5"]
    elif(class_MMI == 'F10'):
        d0, d1, d2, d3, d4, d5 = damage_F_10["0"], damage_F_10["1"], damage_F_10["2"], damage_F_10["3"], damage_F_10["4"], damage_F_10["5"]
    elif(class_MMI == 'F11'):
        d0, d1, d2, d3, d4, d5 = damage_F_11["0"], damage_F_11["1"], damage_F_11["2"], damage_F_11["3"], damage_F_11["4"], damage_F_11["5"]
    return d0, d1, d2, d3, d4, d5
