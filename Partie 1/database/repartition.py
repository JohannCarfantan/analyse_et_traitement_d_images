#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from shutil import copyfile

DB_src = './src/'
DB_train = './train/'
DB_test = './test/'
DB_validation = './validation/'

if not os.path.exists(DB_train):
  os.mkdir(DB_train)
if not os.path.exists(DB_test):
  os.mkdir(DB_test)
if not os.path.exists(DB_validation):
  os.mkdir(DB_validation)

for root, _, files in os.walk(DB_src, topdown=False):
  cls = root.split('/')[-1]
  dir_train = os.path.join(DB_train, cls)
  dir_validation = os.path.join(DB_validation, cls)
  if not os.path.exists(dir_train):
      os.mkdir(dir_train)
  if not os.path.exists(dir_validation):
      os.mkdir(dir_validation)

for root, _, files in os.walk(DB_src, topdown=True):
  cls = root.split('/')[-1]
  for name in files:
    imgSrc = os.path.join(root, name)
    if not name.endswith('.jpg'):
      continue
    imgName = name.split('.')[-2]
    if int(imgName) % 2 == 0:
      imgDst = os.path.join(DB_train, cls, name)
    elif int(imgName) % 3 == 0:
      imgDst = os.path.join(DB_validation, cls, name)
    else:
      imgDst = os.path.join(DB_test, name)
    if not os.path.exists(imgDst):
      copyfile(imgSrc, imgDst)
