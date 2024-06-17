#!/usr/bin/env python
# coding: utf-8

# ### Library Pemetaan dan Operating System

# In[1]:


import os
from pathlib import Path
from PIL import Image
import shutil


# ### Library Plotting dan Analisis Numerik

# In[2]:


import pandas as pd
import numpy as np
import time
import random
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import branca.colormap as cm

from shapely.geometry import Polygon
from shapely.geometry import Point


# ### Library Machine Learning & Deep Learning

# In[3]:


from keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image


# In[4]:


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


# In[5]:


# Pendefinisian Variabel Awal

# 1. Bounding Box Area yang akan dianalisa

# DEFINE POLYGON DAERAH YANG AKAN DIANALISA DISINI
# UPAYAKAN TITIK TITIK POLYGON BERURUTAN SECARA CLOCKWISE / COUNTER CLOCKWISE UNTUK HASIL OPTIMAL
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
padang_poly = Polygon(padang_shape_list)
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

# 2. API Key untuk GCP dan metabase Google Street View Static API
API_KEY = 'INSERT API KEY HERE'
meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
base_url = 'https://maps.googleapis.com/maps/api/streetview?'

# 3. Heading List GSV API (sudut POV pandang GSV yang dipertimbangkan)
heading_list = ["0", "180", "90", "270"]

# 4. Model AI untuk bangunan dan tipologi
model_building_detector = models.load_model('Deep Learning Models/Model B-NB.h5')
model_typology_detector = models.load_model('Deep Learning Models/Typology Classifier/Model RSL - Efficient - D1 - 14.h5')

# 5. Dictionary tipologi bangunan yang dipertimbangkan
typology_dict = {0 : 'Confined Masonry', 1 : 'RC Infilled Masonry',
                2 : 'Timber Structure', 3 : 'Unconfined Masonry'}
typology_acr_dict = {0 : 'CM', 1 : 'RC', 2 : 'TB', 3 : 'UC'}

# 6. Variabel foto_bangunan untuk menyimpan seluruh hasil capture GSV API, koordinat, dan deteksi model
foto_bangunan = []

# 7. Directory tempat saving gambar dan csv
loc_name = 'INPUT (NAMA KOTA/KABUPATEN) HERE'
output_dirs = 'INPUT (OUTPUT PATH DIRECTORY UNTUK GAMBAR/CSV) HERE'


# ## Utility Functions

# In[6]:


# Generating Random Coordinate within a rectangular bounding box. Inputs a list of 2 coordinates
# Outputs a random coordinate within a imaginary rectangular bounding box with a random uniform distribution

def generate_random_coord_rectangle(coords):
    lat_rand = np.random.uniform(coords[0], coords[1], size=None)
    lon_rand = np.random.uniform(coords[2], coords[3], size=None)
    file_name = str(np.round(lat_rand,7)) + ' - ' + str(np.round(lon_rand,7))
    #cache = np.array((lat_rand,lon_rand))
    return lat_rand,lon_rand, str(lat_rand) + ',' + str(lon_rand), file_name


# In[7]:


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


# In[8]:


def preprocess_image(path, target_size = (256,256)):
    img = image.load_img(path, target_size = target_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch
    return img_preprocessed


# In[9]:


def predict_buildings(path, model):
    img_preprocessed = preprocess_image(path, target_size = (256,256))
    prediction = np.squeeze(model.predict(img_preprocessed))
    return prediction


# In[10]:


def predict_typologies(path, model):
    img_preprocessed = preprocess_image(path,target_size = (256,256))
    prediction = np.squeeze(model.predict(img_preprocessed))
    prediction = typology_dict[np.argmax(prediction)]
    print(prediction)
    return prediction


# In[11]:


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


# ## Loading ML Models to Detect Buildings and its Typologies

# In[12]:


def predict_folder(folder):
    for image_path in os.listdir(folder):
        path = folder + image_path
        print(path)
        predict_typology = predict_typologies(path, model_typology_detector)
        foto_bangunan.append([image_path, predict_typology])


# In[13]:


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


# ### Eksekusi GSV API Mining

# In[ ]:


foto_bangunan = sv_mining(padang_poly, loc_name , num_query = 500, min_threshold = 0.65, 
          dirs = output_dirs,radius = 50, size = "256x256")


# In[ ]:


# Extracting data dari list foto_bangunan menjadi csv yang siap digunakan pada database

df = pd.DataFrame(foto_bangunan, columns = ['Index', 'Path', 'Lintang', 'Bujur', 'Heading', 'Tipologi'])
df.to_csv(f'{output_dirs}/{loc_name}.csv', index = False)

