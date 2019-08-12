import requests
from bs4 import BeautifulSoup
import lxml
import os
import urllib
import sys
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import csv
import multiprocessing
import matplotlib.pyplot as plt
###################!!!!!!!data preprocessing!!!!!!!!##############################
#original data size is about 1 million. I only use 5000 because my computer can't bear
train=pd.read_csv('C:/Users/user/Downloads/pythonCode/Landmark/train.csv')
train=train[~train['url'].isin(['None'])]
new_train=train.iloc[0:5000,:]
new_train.index=range(0,len(new_train))

x=0
for i in range(len(new_train)):
	try:
		link=new_train['url'][i]
		local=os.path.join('D:/LandGraph/%s.jpg' % x)
		urllib.request.urlretrieve(link,local) #Download image from url
		x+=1
	except OSError:
		pass
		new_train=new_train.drop(index=i) #Because some image have a problem( 404 not found)
	continue

with open('C:/Users/user/Downloads/pythonCode/Landmark/new_train.csv','w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow(new_train.columns.values.tolist())
	for i in range(0,len(new_train)):
		writer.writerow(new_train.iloc[i,:])
########################################################################################
