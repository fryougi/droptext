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
import matplotlib.pyplot as plt

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

tmpl_0 = cv2.cvtColor(cv2.imread('templates/gray/0.png'), cv2.COLOR_BGR2GRAY)
tmpl_0m = cv2.cvtColor(cv2.imread('templates/gray/0m.png'), cv2.COLOR_BGR2GRAY)
tmpl_1 = cv2.cvtColor(cv2.imread('templates/gray/1.png'), cv2.COLOR_BGR2GRAY)
tmpl_1m = cv2.cvtColor(cv2.imread('templates/gray/1m.png'), cv2.COLOR_BGR2GRAY)
tmpl_2 = cv2.cvtColor(cv2.imread('templates/gray/2.png'), cv2.COLOR_BGR2GRAY)
tmpl_2m = cv2.cvtColor(cv2.imread('templates/gray/2m.png'), cv2.COLOR_BGR2GRAY)
tmpl_3 = cv2.cvtColor(cv2.imread('templates/gray/3.png'), cv2.COLOR_BGR2GRAY)
tmpl_3m = cv2.cvtColor(cv2.imread('templates/gray/3m.png'), cv2.COLOR_BGR2GRAY)
tmpl_4 = cv2.cvtColor(cv2.imread('templates/gray/4.png'), cv2.COLOR_BGR2GRAY)
tmpl_4m = cv2.cvtColor(cv2.imread('templates/gray/4m.png'), cv2.COLOR_BGR2GRAY)
tmpl_5 = cv2.cvtColor(cv2.imread('templates/gray/5.png'), cv2.COLOR_BGR2GRAY)
tmpl_5m = cv2.cvtColor(cv2.imread('templates/gray/5m.png'), cv2.COLOR_BGR2GRAY)
tmpl_6 = cv2.cvtColor(cv2.imread('templates/gray/6.png'), cv2.COLOR_BGR2GRAY)
tmpl_6m = cv2.cvtColor(cv2.imread('templates/gray/6m.png'), cv2.COLOR_BGR2GRAY)
tmpl_7 = cv2.cvtColor(cv2.imread('templates/gray/7.png'), cv2.COLOR_BGR2GRAY)
tmpl_7m = cv2.cvtColor(cv2.imread('templates/gray/7m.png'), cv2.COLOR_BGR2GRAY)
tmpl_8 = cv2.cvtColor(cv2.imread('templates/gray/8.png'), cv2.COLOR_BGR2GRAY)
tmpl_8m = cv2.cvtColor(cv2.imread('templates/gray/8m.png'), cv2.COLOR_BGR2GRAY)
tmpl_9 = cv2.cvtColor(cv2.imread('templates/gray/9.png'), cv2.COLOR_BGR2GRAY)
tmpl_9m = cv2.cvtColor(cv2.imread('templates/gray/9m.png'), cv2.COLOR_BGR2GRAY)
tmpl_x = cv2.cvtColor(cv2.imread('templates/gray/x.png'), cv2.COLOR_BGR2GRAY)
tmpl_xm = cv2.cvtColor(cv2.imread('templates/gray/xm.png'), cv2.COLOR_BGR2GRAY)
tmpl_p = cv2.cvtColor(cv2.imread('templates/gray/p.png'), cv2.COLOR_BGR2GRAY)
tmpl_pm = cv2.cvtColor(cv2.imread('templates/gray/pm.png'), cv2.COLOR_BGR2GRAY)
tmpl_l = cv2.cvtColor(cv2.imread('templates/gray/l.png'), cv2.COLOR_BGR2GRAY)
tmpl_lm = cv2.cvtColor(cv2.imread('templates/gray/lm.png'), cv2.COLOR_BGR2GRAY)
tmpl_r = cv2.cvtColor(cv2.imread('templates/gray/r.png'), cv2.COLOR_BGR2GRAY)
tmpl_rm = cv2.cvtColor(cv2.imread('templates/gray/rm.png'), cv2.COLOR_BGR2GRAY)

tmpl2char = ['0','1','2','3','4','5','6','7','8','9','x','+','(',')']
tmplchars = [tmpl_0,tmpl_1,tmpl_2,tmpl_3,tmpl_4,tmpl_5,tmpl_6,
             tmpl_7,tmpl_8,tmpl_9,tmpl_x,tmpl_p,tmpl_l,tmpl_r]
tmplmasks = [tmpl_0m,tmpl_1m,tmpl_2m,tmpl_3m,tmpl_4m,tmpl_5m,tmpl_6m,
             tmpl_7m,tmpl_8m,tmpl_9m,tmpl_xm,tmpl_pm,tmpl_lm,tmpl_rm]
tmpl_correction = 0.8/np.array([.8,.87,.85,.83,.85,.81,.8,.9,.85,.83,.8,.83,.8,.8])

tmpl_tol = 0.97


def corrtmpl(cvwnd,tmpl,mask):
  h,w,_ = tmpl.shape
  if mask is None:
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED)
  else:
    data = np.zeros((h,w,3),dtype=np.uint8)
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED, data, mask)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  return max_val, max_loc

def corrtmplgray(cvwnd,tmpl,mask):
  if mask is None:
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED)
  else:
    h,w = tmpl.shape
    data = np.zeros((h,w),dtype=np.uint8)
    if cvwnd.shape[1]+2 < h:
      return 0, (0,0)
    res = cv2.matchTemplate(cvwnd, tmpl, cv2.TM_CCORR_NORMED, data, mask)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  return max_val, max_loc

def maxcorrtmpl(cvwnd,templates,masks):
  corr = np.zeros(len(templates))
  xloc = []
  for i in range(len(templates)):
    corr[i], max_loc = corrtmplgray(cvwnd,templates[i],masks[i])
    xloc.append(max_loc[0])
  # Correction factor due to different norms...
  corr = np.multiply(corr, tmpl_correction)
  corrindex = np.argmax(corr)
  #print(corr, xloc)
  # make sure there's a valid match
  if corr[corrindex] < 0.80:
    return -1, -1
  else:
    return corrindex, xloc[corrindex]
  
def matchtext(cvwnd,tmpl2char,tmpls,masks):
  cvparse = np.copy(cvwnd)
  chars = []
  locs = []
  for i in range(7): # sometimes it gets stuck in a loop
    index, loc = maxcorrtmpl(cvparse,tmpls,masks)
    if index == -1:
      break
    else:
      chars.append(tmpl2char[index])
      locs.append(loc)
      # remove the block for next iteration
      cvparse[:,loc:loc+tmpls[index].shape[1]] = np.random.randint(0,255,(14,tmpls[index].shape[1]),dtype=np.uint8)
      #cvparse = np.hstack((cvparse[:,:loc],cvparse[:,loc+tmpls[index].shape[1]:]))
    #print(chars,locs)
    #plt.imshow(cvparse)
  return chars, locs

def matchtextrec(cvwnd,tmpl2char,tmpls,masks):
  # Turn this into a binary search or something...
  for i in range(7): # sometimes it gets stuck in a loop
    index, loc = maxcorrtmpl(cvwnd,tmpls,masks)
    if index == -1:
      return ''
    else:
      # remove the block for next iteration
      #cvparse[:,loc:loc+tmpls[index].shape[1]] = np.random.randint(0,255,(14,tmpls[index].shape[1]),dtype=np.uint8)
      return matchtextrec(cvwnd[:,:loc+1],tmpl2char,tmpls,masks) + tmpl2char[index] + matchtextrec(cvwnd[:,loc-1+tmpls[index].shape[1]:],tmpl2char,tmpls,masks)

time_start = time.perf_counter()

score = 0

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
  # thresholding sucks, use templates with masks instead
  
  #chars, locs = matchtext(cvgray,tmpl2char,tmplchars,tmplmasks)
  #zipped = zip(chars,locs)
  #droptext = ''.join([i for i, j in sorted(zipped, key=lambda t: t[1])])
  #print(droptext)
  droptext = matchtextrec(cvgray,tmpl2char,tmplchars,tmplmasks)
  print(droptext)
  
  if droptext == label:
    score +=1
  
time_end = time.perf_counter()
seconds = time_end - time_start
print("score of {} in {:f}".format(score, seconds))

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