#!/usr/bin/env python3
import json
import os
import numpy as np
import cv2

ROOT = './shared_scene'
for name in ['primary', 'secondary']:
    d = os.path.join(ROOT, name)
    rgb = os.path.join(d, 'latest_color.jpg')
    depth = os.path.join(d, 'latest_depth.npy')
    info = os.path.join(d, 'latest_camera_info.json')
    print(f'== {name} ==')
    if os.path.exists(rgb):
        img = cv2.imread(rgb)
        print('rgb shape:', None if img is None else img.shape)
    else:
        print('rgb missing')
    if os.path.exists(depth):
        arr = np.load(depth)
        print('depth shape:', arr.shape, 'dtype:', arr.dtype)
    else:
        print('depth missing')
    if os.path.exists(info):
        data = json.load(open(info, 'r', encoding='utf-8'))
        print('camera_info width/height:', data.get('width'), data.get('height'))
    else:
        print('camera_info missing')
