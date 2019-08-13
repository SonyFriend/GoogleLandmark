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
###########################################################################
#Original data size is about 1 million. I only use the data which id from 0 to 499, the data size is about 30000.
#Because you need much time to download these images.
train=pd.read_csv('C:/Users/user/Downloads/pythonCode/Landmark/new_train.csv')
train=train[~train['url'].isin(['None'])]
train.index=range(0,len(train))


with open('C:/Users/user/Downloads/pythonCode/Landmark/new_train_id0_499.csv','w',newline='') as f:
	writer=csv.writer(f)
	writer.writerow(train.columns.values.tolist())
	x=0
	for i in range(len(train)):
		try:
			link=train['url'][i]
			local=os.path.join('D:/LandGraph/%s.jpg' % x)
			urllib.request.urlretrieve(link,local) #Downlord images from url.
			writer.writerow(train.iloc[i,:])
			x+=1
		except OSError:
			pass
		continue

########################################################################################
