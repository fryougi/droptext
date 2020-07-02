# -*- coding: utf-8 -*-
"""
Testing drop numbers (grab template locations/windows)

Todo: 
"""

import numpy as np
import cv2
import os, os.path
from PIL import Image,ImageOps
import time

dropfiles = [name for name in os.listdir('data') if os.path.isfile(os.path.join('data',name))]
droplabels = [name.split('.')[0].split(' ')[0] for name in dropfiles]

# Probably want templates for the different drop backgrounds
tmpl_wx = cv2.imread('templates/wx.png')
mask_wx = cv2.imread('templates/wxmask.png')
tmpl_yp = cv2.imread('templates/yp.png')
mask_yp = cv2.imread('templates/ypmask.png')
tmpl_wp = cv2.imread('templates/wp.png')
mask_wp = cv2.imread('templates/wpmask.png')

tmpl_gitem = cv2.imread('templates/gitem.png')
tmpl_sitem = cv2.imread('templates/sitem.png')

tmpl_tol = 0.97

time_start = time.perf_counter()

def corrtmpl(cvwnd,tmpl,mask):
  h,w,_ = tmpl.shape
  if mask is None:
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED)
  else:
    data = np.zeros((h,w,3),dtype=np.uint8)
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED, data, mask)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  return max_val, max_loc

# Load image
for i in range(len(dropfiles)):
  fname = os.path.join('data',dropfiles[i])
  label = droplabels[i]
  
  corr = np.zeros(3)
  locs = []
  
  # Take bottom half of image
  cvimage = cv2.imread(fname)
  cvframe = cvimage[50:,10:]
  # Sometimes it still breaks/confused on the 'Item' text
  # maybe another filter to get the bottom boundary?
  max_val, max_loc = corrtmpl(cvframe, tmpl_gitem, None)
  if max_val > 0.99:
    cvframe = cvframe[:max_loc[1],:]
  max_val, max_loc = corrtmpl(cvframe, tmpl_sitem, None)
  if max_val > 0.99:
    cvframe = cvframe[:max_loc[1],:]
  
  # Go through the different templates
  corr[0], max_loc = corrtmpl(cvframe, tmpl_wx, mask_wx)
  locs.append(max_loc)
  corr[1], max_loc = corrtmpl(cvframe, tmpl_wp, mask_wp)
  locs.append(max_loc)
  corr[2], max_loc = corrtmpl(cvframe, tmpl_yp, mask_yp)
  locs.append(max_loc)
  
  # Figure out why things are breaking
  #if dropfiles[i] == 'x2 (36).png':
  #  print(corr, locs)
  #  break
  
  # get leftmost match
  xmin = 1000
  ymin = 0
  temp_tol = tmpl_tol
  while xmin > 100:
    for j, c in enumerate(corr):
      if j == 2:
        #skip yellow plusses
        continue
      if c > temp_tol:
        if locs[j][0] < xmin:
          xmin = locs[j][0]
          ymin = locs[j][1]
    # there must be a match
    temp_tol -= 0.005
  
  # windowing based on match location
  # probably brittle
  cvwnd = cvframe[ymin-3:ymin+11,xmin+1:-3]
  
  cv2.imwrite(os.path.join('text',dropfiles[i]), cvwnd)
  
  # Try to do some OCR?
  cvhsv = cv2.cvtColor(cvwnd, cv2.COLOR_BGR2HSV)
  lower_gold = np.array([20,50,130])
  upper_gold = np.array([40,255,255])
  mask = cv2.inRange(cvhsv, lower_gold, upper_gold)
  # increase brightness of yellow areas
  cvhsv[:,:,2][mask>0] += np.minimum(250-cvhsv[:,:,2][mask>0], 20)
  cvhsv[:,:,1][mask>0] = 0
  cvrgb = cv2.cvtColor(cvhsv, cv2.COLOR_HSV2BGR)
  pimg = Image.fromarray(cvrgb)
  pgray = ImageOps.grayscale(ImageOps.autocontrast(pimg))
  cvgray = np.array(pgray)
  cv2.imwrite(os.path.join('ocr',dropfiles[i]), cvgray)
  
time_end = time.perf_counter()
seconds = time_end - time_start
print("{:f}".format(seconds))

# List of location windows
# (xtol, ytol)
# (xmin, ymin, xmax, ymax)

# Compute match
#h,w,_ = tmpl.shape
#data = numpy.zeros((h,w,3),dtype=numpy.uint8)
#res = cv2.matchTemplate(cvframe, tmpl, cv2.TM_CCORR_NORMED, data, mask)
#res = cv2.matchTemplate(cvframe, tmpl, cv2.TM_CCORR_NORMED)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# Get box
#top_left = max_loc
#bottom_right = (top_left[0]+w, top_left[1]+h)
#cv2.rectangle(cvframe, top_left, bottom_right, 255, 2)
#rect = (top_left[0],top_left[1],bottom_right[0],bottom_right[1])
#print(rect)
#print(max_val)
# Display
#cv2.imshow('cvimage', cvframe)
#cv2.waitKey(0)
#cv2.destroyAllWindows()