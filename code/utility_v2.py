#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil
from ultralytics import YOLO

import pyproj
from shapely.geometry import Polygon
from shapely.geometry import Point

import pandas as pd
import numpy as np
import time
import random
import requests


# In[20]:


# This function calculates azimuth and distances in meters, given the information of initial coordinates (lat1, lon1)
# and terminus coordinate (lat2, lon2)

def calc_azimuth_distance(lat1, long1, lat2, long2):
    fwd_azimuth,back_azimuth,distance = geodesic.inv(long1, lat1, long2, lat2)
    return fwd_azimuth, distance

geodesic = pyproj.Geod(ellps='WGS84')


# In[21]:


def calc_PGA_Young(mag, depth, epic_dist, hyp_dist):
    c1, c2 = 0, 0
    c3, c4, c5 = -2.552, 1.45, -0.1
    zt = 1
    ln_pga = 0.2418 + 1.414*mag + c1 + c2*(10-mag)**3 + c3*np.log(epic_dist + 1.7818*np.exp(0.544*mag)) + 0.00607*depth + 0.3846*zt
    pga = np.exp(ln_pga)
    return pga


# In[22]:


# Based on Table 10 of SNI 1726-2019 of Earthquake-Resisting Buildings
def calc_PGA_surface(pga, site_level = 'SE'):
    if(site_level == 'SA'):
        surface_pga = 0.8 * pga
    elif(site_level == 'SB'):
        surface_pga = 0.9 * pga
    elif(site_level == 'SC'):
        if(pga < 0.1):
            surface_pga = 1.3 * pga
        elif(pga > 0.6):
            surface_pga = 1.2 * pga
        else:
            xp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            yp = [1.3, 1.2, 1.2, 1.2, 1.2, 1.2]
            surface_pga = np.interp(pga, xp, yp) * pga
    elif(site_level == 'SD'):
        if(pga < 0.1):
            surface_pga = 1.6 * pga
        elif(pga > 0.6):
            surface_pga = 1.4 * pga
        else:
            xp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            yp = [1.6, 1.4, 1.3, 1.2, 1.1, 1.1]
            surface_pga = np.interp(pga, xp, yp) * pga
    elif(site_level == 'SE'):
        if(pga < 0.1):
            surface_pga = 2.4 * pga
        elif(pga > 0.6):
            surface_pga = 1.1 * pga
        else:
            xp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            yp = [2.4, 1.9, 1.6, 1.4, 1.2, 1.1]
            surface_pga = np.interp(pga, xp, yp) * pga
            print(np.interp(pga, xp, yp))
    return surface_pga


# In[23]:


g_to_gal = 981
def calc_MMI(surface_pga):
    s_pga_gal = surface_pga * g_to_gal
    mmi = 0.1417 + 3.2335*np.log10(s_pga_gal)
    return mmi


# In[24]:


def randomize_building_vulnerability_class(building_typology):
    if(building_typology == 'Confined Masonry'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.27, 0.64, 0.09])[0]
    elif(building_typology == 'RC Infilled Masonry'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.27, 0.64, 0.09])[0]
    elif(building_typology == 'Timber Structure'):
        kelas =  np.random.choice(['B','C','D'], 1, p=[0.09, 0.64, 0.27])[0]
    elif(building_typology == 'Unconfined Masonry'):
        kelas =  np.random.choice(['A','B','C'], 1, p=[0.09, 0.82, 0.09])[0]
    elif(building_typology == 'Moment Resisting Frame'):
        kelas =  np.random.choice(['C','D','E','F'], 1, p=[0.09, 0.27, 0.37, 0.27])[0]
    return kelas


# In[25]:


# Generating Random Coordinate within a rectangular bounding box. Inputs a list of 2 coordinates
# Outputs a random coordinate within a imaginary rectangular bounding box with a random uniform distribution

def generate_random_coord_rectangle(coords):
    lat_rand = np.random.uniform(coords[0], coords[1], size=None)
    lon_rand = np.random.uniform(coords[2], coords[3], size=None)
    file_name = str(np.round(lat_rand,7)) + ' - ' + str(np.round(lon_rand,7))
    #cache = np.array((lat_rand,lon_rand))
    return lat_rand,lon_rand, str(lat_rand) + ',' + str(lon_rand), file_name


# In[26]:


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


# In[27]:


def draw_bboxes(image_path, detections, output_path=None, show_image=False):
    # Class dictionary and associated colors
    class_dict = {
        0: 'Confined Masonry', 1: 'RC Infilled Masonry',2: 'Timber Structure',
        3: 'Unconfined Masonry', 4: 'Moment Resisting Frame'
    }

    class_colors = {
        0: (255, 0, 0),       # Red
        1: (0, 255, 0),       # Green
        2: (0, 0, 255),       # Blue
        3: (255, 165, 0),     # Orange
        4: (128, 0, 128)      # Purple
    }

    # Load and prepare the image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_width, image_height = image.size
    draw = ImageDraw.Draw(image)
    
    # Load font (fallback to default if DejaVuSans is not available)
    try:
        font_path = r"C:\Users\hafid\Python Files\TA & Thesis\Thesis\Main Work\font\DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, size=12)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw boxes
    for cls_id, cx, cy, w, h in detections:
        x1 = int((cx - w / 2) * image_width)
        y1 = int((cy - h / 2) * image_height)
        x2 = int((cx + w / 2) * image_width)
        y2 = int((cy + h / 2) * image_height)

        label = class_dict.get(int(cls_id), "Unknown")
        color = class_colors.get(int(cls_id), (255, 255, 255))

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Text label with background
        text = label
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_x = x1
        text_y = max(y1 - text_height - 4, 0)

        # Draw background rectangle
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
            fill=color
        )
        # Draw text
        draw.text((text_x + 2, text_y + 2), text, fill="white", font=font)

    # Save or show the image
    if output_path:
        image.save(output_path)
    if show_image:
        image.show()

    return image


# In[28]:


def predict_typologies(path, model):
    results = model(path, iou = 0.3, agnostic_nms = True, stream = True)
    valid_results = []
    for r in results:
        typology = r.boxes.cls
        box_xywhn = r.boxes.xywhn
        print(typology, box_xywhn)
        # Filtering the valid bbox to export to Excel database
        # The valid bbox have to:
        # 1. The x,y normalized centerline should not be (< 0.1 or > 0.9 image width and height)
        # discarding the detections in the extreme ends of the image
        # 2. The w * h (width and height of the bbox) have a minimum of 8% of the image area.
        # Less than 8% indicates the bbox is too small (indicating the building is too far away)
        for t, box in zip(typology, box_xywhn):
            x_center, y_center = box[0], box[1]
            w, h = box[2], box[3]
            if(x_center > 0.1 and x_center < 0.9 and y_center > 0.1 and y_center < 0.9):
                if(w * h > 0.08):
                    valid_results.append([t.tolist(), x_center.tolist(), y_center.tolist(), w.tolist(), h.tolist()])
    return valid_results

# In[29]:
def get_street_view_image(lat, lon, api, heading=0, pitch=0, fov=90, image_size='640x640', save_path='street_view.jpg'):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    params = {
        "size": image_size,          # max size 640x640 for free tier
        "location": f"{lat},{lon}",
        "heading": heading,          # camera direction (0 = north, 90 = east, etc.)
        "pitch": pitch,              # up/down angle (-90 to 90)
        "fov": fov,                  # field of view (10 to 120)
        "key": API_KEY
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Street View image saved to {save_path}")
        response.close()
        return True
    else:
        print(f"Failed to retrieve image. Status code: {response.status_code}")
        print(response.text)
        return False

# In[30]:


def predict_folder(folder):
    for image_path in os.listdir(folder):
        path = folder + image_path
        print(path)
        predict_typology = predict_typologies(path, model_typology_detector)
        foto_bangunan.append([image_path, predict_typology])


# In[31]:


# 2. API Key untuk GCP dan metabase Google Street View Static API
API_KEY = 'enter your own API KEY here'
meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
base_url = 'https://maps.googleapis.com/maps/api/streetview?'

# 3. Heading List GSV API (sudut POV pandang GSV yang dipertimbangkan)
heading_list = ["0", "180", "90", "270"]

# 4. Model AI untuk bangunan dan tipologi
typology_detector_path = r"C:\Users\hafid\Python Files\TA & Thesis\Thesis\Main Work\Model_OD\typology_v.3.0.pt"
model_typology_detector = YOLO(typology_detector_path)

# 5. Dictionary tipologi bangunan yang dipertimbangkan
typology_dict = {0 : 'Confined Masonry', 1 : 'RC Infilled Masonry',
                2 : 'Timber Structure', 3 : 'Unconfined Masonry', 4 : 'Moment Resisting Frame'}

# 6. Variabel foto_bangunan untuk menyimpan seluruh hasil capture GSV API, koordinat, dan deteksi model
foto_bangunan = []

def sv_mining(poly, loc_name, num_query,  dirs = None,
             radius = 50, size = "512x512"):
    not_found = 0
    dapat = 0
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print("Directory created successfully!")
    coords = random_points_in_polygon(poly, num_query)
    print(coords)
    for cnt, i in enumerate(coords):
        lat, lon = i[0], i[1]
        file_name = str(np.round(float(lat),7)) + ' - ' + str(np.round(float(lon),7))
        
        img_path = dirs + str(cnt) + ' ' + loc_name + ' ' + file_name + '.jpg'
        out_path = f"{img_path.replace('.jpg','-output.jpg')}"
            
        ok = get_street_view_image(lat, lon, API_KEY, heading=0, pitch=0, fov=90, 
                              image_size='640x640', save_path=img_path)
        if(ok):
            if((cnt+1) % 50 == 0):
                print('Pencarian gambar ke - ' + str(cnt+1))
            predict_typology = predict_typologies(img_path, model_typology_detector)
            if(predict_typology == []):
                os.remove(img_path)
                not_found+= 1
                continue
            else:
                predict_image = draw_bboxes(img_path, predict_typology, out_path)
                for i in predict_typology:
                    foto_bangunan.append([cnt, img_path, lat, lon, typology_dict[i[0]]])
                dapat+= 1
        else:
            not_found+= 1
    print("Total gambar tidak mampu di-query : "+ str(not_found) + " gambar")
    return foto_bangunan


# In[32]:


def calculate_dmg_prob(class_MMI):
    none, nearly_none, few, many, most, nearly_all, alls = 0.0, 0.015, 0.09, 0.35, 0.745, 0.97, 1.00
    
    #Dictionary Damage Class A-F, MMI 5
    damage_A_5 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_B_5 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_C_5 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_D_5 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_E_5 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_F_5 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 6
    damage_A_6 = {'0' : many+7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    damage_B_6 = {'0' : many+7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    damage_C_6 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_D_6 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_E_6 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_F_6 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 7
    damage_A_7 = {'0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none}
    damage_B_7 = {'0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none}
    damage_C_7 = {'0' : many+7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    damage_D_7 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_E_7 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_F_7 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 8
    damage_A_8 = {'0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few}
    damage_B_8 = {'0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none}
    damage_C_8 = {'0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none}
    damage_D_8 = {'0' : many + 7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    damage_E_8 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    damage_F_8 = {'0' : alls, '1' : none, '2' : none, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 9
    damage_A_9 = {'0' : none, '1' : none, '2' : 1/3*few, '3' : 3*few, '4' : many, '5' : many}
    damage_B_9 = {'0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few}
    damage_C_9 = {'0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none}
    damage_D_9 = {'0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none}
    damage_E_9 = {'0' : many + 7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    damage_F_9 = {'0' : alls-few, '1' : few, '2' : none, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 10
    damage_A_10 = {'0' : none, '1' : none, '2' : none, '3' : 5/6*few, '4' : 2*few, '5' : most}
    damage_B_10 = {'0' : none, '1' : none, '2' : 1/3*few, '3' : 2*few, '4' : many+few, '5' : many}
    damage_C_10 = {'0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few}
    damage_D_10 = {'0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none}
    damage_E_10 = {'0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none}
    damage_F_10 = {'0' : many+7/3*few, '1' : many, '2' : few, '3' : none, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 11
    damage_A_11 = {'0' : none, '1' : none, '2' : none, '3' : none, '4' : 5/6*few, '5' : most+2*few}
    damage_B_11 = {'0' : none, '1' : none, '2' : none, '3' : nearly_none, '4' : 8/3*few, '5' : most}
    damage_C_11 = {'0' : none, '1' : none, '2' : none, '3' : 4/3*few, '4' : many+2*few, '5' : many}
    damage_D_11 = {'0' : none, '1' : 1/3*few, '2' : 2*few, '3' : many, '4' : many, '5' : few}
    damage_E_11 = {'0' : 1/3*few, '1' : 2*few, '2' : many, '3' : many, '4' : few, '5' : none}
    damage_F_11 = {'0' : 7/3*few, '1' : many, '2' : many, '3' : few, '4' : none, '5' : none}
    
    #Dictionary Damage Class A-F, MMI 12
    damage_A_12 = {'0' : none, '1' : none, '2' : none, '3' : none, '4' : none, '5' : alls}
    damage_B_12 = {'0' : none, '1' : none, '2' : none, '3' :  none, '4' : none, '5' : alls}
    damage_C_12 = {'0' : none, '1' : none, '2' : none, '3' : none, '4' : 1/3*few, '5' : nearly_all}
    damage_D_12 = {'0' : none, '1' : none, '2' : 1/3*few, '3' : 1/2*few, '4' : 2*few, '5' : most}
    damage_E_12 = {'0' : none, '1' : nearly_none, '2' : 2/3*few, '3' : few, '4' : 2*few, '5' : most-few}
    damage_F_12 = {'0' : none, '1' : 1/3*few, '2' : few, '3' : few, '4' : many, '5' : many+few}
    
    if(class_MMI == 'A5'):
        d0, d1, d2, d3, d4, d5 = damage_A_5["0"], damage_A_5["1"], damage_A_5["2"], damage_A_5["3"], damage_A_5["4"], damage_A_5["5"]
    elif(class_MMI == 'A6'):
        d0, d1, d2, d3, d4, d5 = damage_A_6["0"], damage_A_6["1"], damage_A_6["2"], damage_A_6["3"], damage_A_6["4"], damage_A_6["5"]
    elif(class_MMI == 'A7'):
        d0, d1, d2, d3, d4, d5 = damage_A_7["0"], damage_A_7["1"], damage_A_7["2"], damage_A_7["3"], damage_A_7["4"], damage_A_7["5"]
    elif(class_MMI == 'A8'):
        d0, d1, d2, d3, d4, d5 = damage_A_8["0"], damage_A_8["1"], damage_A_8["2"], damage_A_8["3"], damage_A_8["4"], damage_A_8["5"]
    elif(class_MMI == 'A9'):
        d0, d1, d2, d3, d4, d5 = damage_A_9["0"], damage_A_9["1"], damage_A_9["2"], damage_A_9["3"], damage_A_9["4"], damage_A_9["5"]
    elif(class_MMI == 'A10'):
        d0, d1, d2, d3, d4, d5 = damage_A_10["0"], damage_A_10["1"], damage_A_10["2"], damage_A_10["3"], damage_A_10["4"], damage_A_10["5"]
    elif(class_MMI == 'A11'):
        d0, d1, d2, d3, d4, d5 = damage_A_11["0"], damage_A_11["1"], damage_A_11["2"], damage_A_11["3"], damage_A_11["4"], damage_A_11["5"]
    elif(class_MMI == 'A12'):
        d0, d1, d2, d3, d4, d5 = damage_A_12["0"], damage_A_12["1"], damage_A_12["2"], damage_A_12["3"], damage_A_12["4"], damage_A_12["5"]
    elif(class_MMI == 'B5'):
        d0, d1, d2, d3, d4, d5 = damage_B_5["0"], damage_B_5["1"], damage_B_5["2"], damage_B_5["3"], damage_B_5["4"], damage_B_5["5"]
    elif(class_MMI == 'B6'):
        d0, d1, d2, d3, d4, d5 = damage_B_6["0"], damage_B_6["1"], damage_B_6["2"], damage_B_6["3"], damage_B_6["4"], damage_B_6["5"]
    elif(class_MMI == 'B7'):
        d0, d1, d2, d3, d4, d5 = damage_B_7["0"], damage_B_7["1"], damage_B_7["2"], damage_B_7["3"], damage_B_7["4"], damage_B_7["5"]
    elif(class_MMI == 'B8'):
        d0, d1, d2, d3, d4, d5 = damage_B_8["0"], damage_B_8["1"], damage_B_8["2"], damage_B_8["3"], damage_B_8["4"], damage_B_8["5"]
    elif(class_MMI == 'B9'):
        d0, d1, d2, d3, d4, d5 = damage_B_9["0"], damage_B_9["1"], damage_B_9["2"], damage_B_9["3"], damage_B_9["4"], damage_B_9["5"]
    elif(class_MMI == 'B10'):
        d0, d1, d2, d3, d4, d5 = damage_B_10["0"], damage_B_10["1"], damage_B_10["2"], damage_B_10["3"], damage_B_10["4"], damage_B_10["5"]
    elif(class_MMI == 'B11'):
        d0, d1, d2, d3, d4, d5 = damage_B_11["0"], damage_B_11["1"], damage_B_11["2"], damage_B_11["3"], damage_B_11["4"], damage_B_11["5"]
    elif(class_MMI == 'B12'):
        d0, d1, d2, d3, d4, d5 = damage_B_12["0"], damage_B_12["1"], damage_B_12["2"], damage_B_12["3"], damage_B_12["4"], damage_B_12["5"]
    elif(class_MMI == 'C5'):
        d0, d1, d2, d3, d4, d5 = damage_C_5["0"], damage_C_5["1"], damage_C_5["2"], damage_C_5["3"], damage_C_5["4"], damage_C_5["5"]
    elif(class_MMI == 'C6'):
        d0, d1, d2, d3, d4, d5 = damage_C_6["0"], damage_C_6["1"], damage_C_6["2"], damage_C_6["3"], damage_C_6["4"], damage_C_6["5"]
    elif(class_MMI == 'C7'):
        d0, d1, d2, d3, d4, d5 = damage_C_7["0"], damage_C_7["1"], damage_C_7["2"], damage_C_7["3"], damage_C_7["4"], damage_C_7["5"]
    elif(class_MMI == 'C8'):
        d0, d1, d2, d3, d4, d5 = damage_C_8["0"], damage_C_8["1"], damage_C_8["2"], damage_C_8["3"], damage_C_8["4"], damage_C_8["5"]
    elif(class_MMI == 'C9'):
        d0, d1, d2, d3, d4, d5 = damage_C_9["0"], damage_C_9["1"], damage_C_9["2"], damage_C_9["3"], damage_C_9["4"], damage_C_9["5"]
    elif(class_MMI == 'C10'):
        d0, d1, d2, d3, d4, d5 = damage_C_10["0"], damage_C_10["1"], damage_C_10["2"], damage_C_10["3"], damage_C_10["4"], damage_C_10["5"]
    elif(class_MMI == 'C11'):
        d0, d1, d2, d3, d4, d5 = damage_C_11["0"], damage_C_11["1"], damage_C_11["2"], damage_C_11["3"], damage_C_11["4"], damage_C_11["5"]
    elif(class_MMI == 'C12'):
        d0, d1, d2, d3, d4, d5 = damage_C_12["0"], damage_C_12["1"], damage_C_12["2"], damage_C_12["3"], damage_C_12["4"], damage_C_12["5"]
    elif(class_MMI == 'D5'):
        d0, d1, d2, d3, d4, d5 = damage_D_5["0"], damage_D_5["1"], damage_D_5["2"], damage_D_5["3"], damage_D_5["4"], damage_D_5["5"]
    elif(class_MMI == 'D6'):
        d0, d1, d2, d3, d4, d5 = damage_D_6["0"], damage_D_6["1"], damage_D_6["2"], damage_D_6["3"], damage_D_6["4"], damage_D_6["5"]
    elif(class_MMI == 'D7'):
        d0, d1, d2, d3, d4, d5 = damage_D_7["0"], damage_D_7["1"], damage_D_7["2"], damage_D_7["3"], damage_D_7["4"], damage_D_7["5"]
    elif(class_MMI == 'D8'):
        d0, d1, d2, d3, d4, d5 = damage_D_8["0"], damage_D_8["1"], damage_D_8["2"], damage_D_8["3"], damage_D_8["4"], damage_D_8["5"]
    elif(class_MMI == 'D9'):
        d0, d1, d2, d3, d4, d5 = damage_D_9["0"], damage_D_9["1"], damage_D_9["2"], damage_D_9["3"], damage_D_9["4"], damage_D_9["5"]
    elif(class_MMI == 'D10'):
        d0, d1, d2, d3, d4, d5 = damage_D_10["0"], damage_D_10["1"], damage_D_10["2"], damage_D_10["3"], damage_D_10["4"], damage_D_10["5"]
    elif(class_MMI == 'D11'):
        d0, d1, d2, d3, d4, d5 = damage_D_11["0"], damage_D_11["1"], damage_D_11["2"], damage_D_11["3"], damage_D_11["4"], damage_D_11["5"]
    elif(class_MMI == 'D12'):
        d0, d1, d2, d3, d4, d5 = damage_D_12["0"], damage_D_12["1"], damage_D_12["2"], damage_D_12["3"], damage_D_12["4"], damage_D_12["5"]
    elif(class_MMI == 'E5'):
        d0, d1, d2, d3, d4, d5 = damage_E_5["0"], damage_E_5["1"], damage_E_5["2"], damage_E_5["3"], damage_E_5["4"], damage_E_5["5"]
    elif(class_MMI == 'E6'):
        d0, d1, d2, d3, d4, d5 = damage_E_6["0"], damage_E_6["1"], damage_E_6["2"], damage_E_6["3"], damage_E_6["4"], damage_E_6["5"]
    elif(class_MMI == 'E7'):
        d0, d1, d2, d3, d4, d5 = damage_E_7["0"], damage_E_7["1"], damage_E_7["2"], damage_E_7["3"], damage_E_7["4"], damage_E_7["5"]
    elif(class_MMI == 'E8'):
        d0, d1, d2, d3, d4, d5 = damage_E_8["0"], damage_E_8["1"], damage_E_8["2"], damage_E_8["3"], damage_E_8["4"], damage_E_8["5"]
    elif(class_MMI == 'E9'):
        d0, d1, d2, d3, d4, d5 = damage_E_9["0"], damage_E_9["1"], damage_E_9["2"], damage_E_9["3"], damage_E_9["4"], damage_E_9["5"]
    elif(class_MMI == 'E10'):
        d0, d1, d2, d3, d4, d5 = damage_E_10["0"], damage_E_10["1"], damage_E_10["2"], damage_E_10["3"], damage_E_10["4"], damage_E_10["5"]
    elif(class_MMI == 'E11'):
        d0, d1, d2, d3, d4, d5 = damage_E_11["0"], damage_E_11["1"], damage_E_11["2"], damage_E_11["3"], damage_E_11["4"], damage_E_11["5"]
    elif(class_MMI == 'E12'):
        d0, d1, d2, d3, d4, d5 = damage_E_12["0"], damage_E_12["1"], damage_E_12["2"], damage_E_12["3"], damage_E_12["4"], damage_E_12["5"]
    elif(class_MMI == 'F5'):
        d0, d1, d2, d3, d4, d5 = damage_F_5["0"], damage_F_5["1"], damage_F_5["2"], damage_F_5["3"], damage_F_5["4"], damage_F_5["5"]
    elif(class_MMI == 'F6'):
        d0, d1, d2, d3, d4, d5 = damage_F_6["0"], damage_F_6["1"], damage_F_6["2"], damage_F_6["3"], damage_F_6["4"], damage_F_6["5"]
    elif(class_MMI == 'F7'):
        d0, d1, d2, d3, d4, d5 = damage_E_7["0"], damage_F_7["1"], damage_F_7["2"], damage_F_7["3"], damage_F_7["4"], damage_F_7["5"]
    elif(class_MMI == 'F8'):
        d0, d1, d2, d3, d4, d5 = damage_F_8["0"], damage_F_8["1"], damage_F_8["2"], damage_F_8["3"], damage_F_8["4"], damage_F_8["5"]
    elif(class_MMI == 'F9'):
        d0, d1, d2, d3, d4, d5 = damage_F_9["0"], damage_F_9["1"], damage_F_9["2"], damage_F_9["3"], damage_F_9["4"], damage_F_9["5"]
    elif(class_MMI == 'F10'):
        d0, d1, d2, d3, d4, d5 = damage_F_10["0"], damage_F_10["1"], damage_F_10["2"], damage_F_10["3"], damage_F_10["4"], damage_F_10["5"]
    elif(class_MMI == 'F11'):
        d0, d1, d2, d3, d4, d5 = damage_F_11["0"], damage_F_11["1"], damage_F_11["2"], damage_F_11["3"], damage_F_11["4"], damage_F_11["5"]
    elif(class_MMI == 'F12'):
        d0, d1, d2, d3, d4, d5 = damage_F_12["0"], damage_F_12["1"], damage_F_12["2"], damage_F_12["3"], damage_F_12["4"], damage_F_12["5"]
    return d0, d1, d2, d3, d4, d5

