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
#original data size is about 1 million. I only use the data which landmark_id is from 0 to 1000, the data size is about 56000. 
train=pd.read_csv('C:/Users/user/Downloads/pythonCode/Landmark/train.csv')
train=train[~train['url'].isin(['None'])]
train.index=range(0,len(train))
x=0
for i in range(len(train)):
	try:
		link=train['url'][i]
		local=os.path.join('D:/LandGraph/%s.jpg' % x)
		urllib.request.urlretrieve(link,local) #Dowmlad image from url
		x+=1
	except OSError:
		pass
		train=train.drop(index=i) #some url have a problem(404 not found)
	continue

with open('C:/Users/user/Downloads/pythonCode/Landmark/new_train_id1_1000.csv','w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow(train.columns.values.tolist())
	for i in range(0,len(train)):
		writer.writerow(train.iloc[i,:])
########################################################################################
