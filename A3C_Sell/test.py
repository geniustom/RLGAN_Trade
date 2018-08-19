import numpy as np
import sys

import matplotlib.pyplot as plt
import pygame

import skimage
from skimage import transform, color, exposure

import keras

from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K

import TradeGame

BETA = 0.01
const = 1e-5
win = 0
lose = 0
points = 0
cnt = 0
lose_cnt=[]
win_cnt=[]
point_list=[]
profit=[]


#loss function for policy output
def logloss(y_true, y_pred):     #policy loss
	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
	return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (160,300), mode = 'constant')
	#image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	#plt.imshow(image, cmap ='gray'); plt.show();
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	return image

model = load_model("saved_models/model_updates_2d", custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
Trade = TradeGame.Game(mode='test')

currentScore = 0
topScore = 0
a_t = [1,0]
FIRST_FRAME = True

terminal = False
r_t = 0
while True:
	if FIRST_FRAME:
		x_t, r_t, terminal = Trade.step(a_t)
		x_t = preprocess(x_t)
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
		FIRST_FRAME = False		
	else:
		x_t, r_t, terminal = Trade.step(a_t)
		x_t = preprocess(x_t)
		s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
	
	y = model.predict(s_t)[0]
	a_t=Trade.gen_action(y)
	#no = np.random.random()	
	#no = np.random.rand()
	#a_t = [0,1] if no < y[0] else [1,0]    #stochastic policy
	#a_t = [0,1] if 0.5 <y[0] else [1,0]   #deterministic policy
	

	if terminal == True:
		FIRST_FRAME = True
		terminal = False
		currentScore = 0
	
		points+=r_t
		profit.append(points)
		if r_t>0: win+=1
		elif r_t<0: lose+=1
	
	if Trade.DateIndex+1==len(Trade.DateList): break

print("win:",win,"lose:",lose,"total points:",points)
plt.plot(profit,"c");plt.savefig("profit_3.png");plt.show();


#-------------- code for checking performance of saved models by finding average scores for 10 runs------------------

# for i in range(1,6):
# 	model = load_model("model" + str(i), custom_objects={'binarycrossentropy': bce, 'sumofsquares': ss})
# 	score = 0
# 	counter = 0
# 	while counter<10:	
# 		x_t, r_t, terminal = Trade.frame_step(a_t)

# 		score += 1
# 		if r_t == -1:
# 			counter += 1
 	
# 		x_t = preprocess(x_t)

# 		if FIRST_FRAME:
# 			s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
			
# 		else:
# 			s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

# 		y = model.predict(s_t)
# 		no = np.random.random()
		
# 		print(y)
# 		if FIRST_FRAME:
# 			a_t = [0,1]
# 			FIRST_FRAME = False
# 		else:
# 			no = np.random.rand()
# 			a_t = [0,1] if no < y[0] else [1,0]	
# 			#a_t = [0,1] if 0.5 <y[0] else [1,0]
		
# 		if r_t == -1:
# 			FIRST_FRAME = True

# 	f = open("test_rewards.txt","a")
# 	f.write(str(score/10)+ "\n")
# 	f.close()
