import numpy as np


import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten,Input,Dropout
from keras.layers.convolutional import Convolution2D,Conv1D
from keras.optimizers import RMSprop,Adam
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

import pygame
import threading
import TradeGame
import matplotlib.pyplot as plt

GAMMA = 0.99                #discount value
BETA = 0.01                 #regularisation coefficient
IMAGE_ROWS = 160
IMAGE_COLS = 300
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4
EPISODE = 0
THREADS = 2
const = 1e-5

# Step 0: Define report
win = 0
lose = 0
points = 0
cnt = 0
lose_cnt=[]
win_cnt=[]
point_list=[]
profit=[]

episode_r = []
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_output = []
episode_critic = []

ACTIONS = 2
a_t = np.zeros(ACTIONS)

#loss function for policy output
def logloss(y_true, y_pred):     #policy loss
	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1) 
	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
	return K.sum(K.square(y_pred - y_true), axis=-1)

#function buildmodel() to define the structure of the neural network in use 
def buildmodel():
	print("Model building begins")

	model = Sequential()
	keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	h0 = Convolution2D(64, 3, 3, subsample=(4, 4), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(S)
	h1 = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h0)
	h2 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h1)
	h3 = Flatten()(h2)
	h4 = Dropout(0.2)(h3)
	h5 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h4)
	h6 = Dropout(0.2)(h5)
	P = Dense(1, name = 'o_P', activation = 'sigmoid', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h6)
	V = Dense(1, name = 'o_V', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h6)

	model = Model(inputs = S, outputs = [P,V])
	opt = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss = {'o_P': logloss, 'o_V': sumofsquares}, loss_weights = {'o_P': 1., 'o_V' : 0.5}, optimizer = opt)
	return model

#function to preprocess an image before giving as input to the neural network
def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (160,300), mode = 'constant')
	#image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	#plt.imshow(image, cmap ='gray'); plt.show();
	image = image.reshape(1, image.shape[0], image.shape[1], 1)
	#pygame.event.get()  #讓遊戲畫面能夠更新
	return image


def runprocess(thread_id, s_t):
	global model
	global cnt,points,profit,win,lose,lose_cnt,win_cnt,point_list


	terminal = False
	r_t = 0
	r_store = []
	state_store = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	output_store = []
	critic_store = []
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

	while terminal==False:
		with graph.as_default():
			y = model.predict(s_t)[0]

		no = np.random.rand()
		a_t = [0,1] if no < y else [1,0]  #stochastic action
		#a_t = [0,1] if 0.5 <y[0] else [1,0]  #deterministic action
		#a_t=GameTread[thread_id].gen_action(y)

		x_t, r_t, terminal = GameTread[thread_id].step(a_t)
		x_t = preprocess(x_t)

		with graph.as_default():
			critic_reward = model.predict(s_t)[1]

		y = 0 if a_t[0] == 1 else 1

		r_store = np.append(r_store, r_t)
		state_store = np.append(state_store, s_t, axis = 0)
		output_store = np.append(output_store, y)
		critic_store = np.append(critic_store, critic_reward)
		
		s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
		#print("Frame = " + str(T) + ", Updates = " + str(EPISODE) + ", Thread = " + str(thread_id) + ", Output = "+ str(intermediate_output))
		#pygame.event.get()  #讓遊戲畫面能夠更新
	
		########################  統計輸出報表用  ########################
		cnt+=1
		if terminal==True:
			points+=r_t
			profit.append(points)
			if r_t>0: win+=1
			elif r_t<0: lose+=1
	
		########################  統計輸出報表用  ########################		
	
	if terminal == False:
		r_store[len(r_store)-1] = critic_store[len(r_store)-1]
	else:
		r_store[len(r_store)-1] = -1
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
	
	for i in range(2,len(r_store)+1):
		r_store[len(r_store)-i] = r_store[len(r_store)-i] + GAMMA*r_store[len(r_store)-i + 1]

	if GameTread[thread_id].DateIndex == 0 and lose!=0 and win!=0:
		lose_cnt.append(lose)
		win_cnt.append(win)
		point_list.append(points)
		plt.plot(lose_cnt,"g");plt.plot(win_cnt,"r");plt.savefig("profit_1.png");plt.show();
		plt.plot(point_list,"b");plt.savefig("profit_2.png");plt.show();
		plt.plot(profit,"c");plt.savefig("profit_3.png");plt.show();
		print("run:",cnt,"date:",GameTread[thread_id].Date ,"profit:",points)
		lose=0
		win=0
		points=0
		profit=[]	

	return s_t, state_store, output_store, r_store, critic_store

#function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def step_decay(epoch):
	decay = 3.2e-8
	lrate = LEARNING_RATE - epoch*decay
	lrate = max(lrate, 0)
	return lrate

class actorthread(threading.Thread):
	def __init__(self,thread_id, s_t):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.next_state = s_t

	def run(self):
		global episode_output
		global episode_r
		global episode_critic
		global episode_state

		threadLock.acquire()
		self.next_state, state_store, output_store, r_store, critic_store = runprocess(self.thread_id, self.next_state)
		self.next_state = self.next_state.reshape(self.next_state.shape[1], self.next_state.shape[2], self.next_state.shape[3])

		episode_r = np.append(episode_r, r_store)
		episode_output = np.append(episode_output, output_store)
		episode_state = np.append(episode_state, state_store, axis = 0)
		episode_critic = np.append(episode_critic, critic_store)

		threadLock.release()



# initialize a new model using buildmodel() or use load_model to resume training an already trained model
#model = buildmodel()
model = load_model("saved_models/model_updates_3050", custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
#plot_model(model, to_file='model.png')
model._make_predict_function()
graph = tf.get_default_graph()

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('o_P').output)


a_t[0] = 1 #index 0 = no flap, 1= flap
#output of network represents probability of flap

GameTread = []
for i in range(0,THREADS):
	GameTread.append(TradeGame.Game())

states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

#initializing state of each thread
for i in range(0, len(GameTread)):
	image = GameTread[i].getCurrentFrame()
	image = preprocess(image)
	state = np.concatenate((image, image, image, image), axis=3)
	states = np.append(states, state, axis = 0)

while True:	
	threadLock = threading.Lock()
	threads = []
	for i in range(0,THREADS):
		threads.append(actorthread(i,states[i]))

	states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

	for i in range(0,THREADS):
		threads[i].start()

	#thread.join() ensures that all threads fininsh execution before proceeding further
	for i in range(0,THREADS):
		threads[i].join()

	for i in range(0,THREADS):
		state = threads[i].next_state
		state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
		states = np.append(states, state, axis = 0)

	e_mean = np.mean(episode_r)
	#advantage calculation for each action taken
	advantage = episode_r - episode_critic

	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [lrate]

	weights = {'o_P':advantage, 'o_V':np.ones(len(advantage))}	
	#backpropagation
	history = model.fit(episode_state, [episode_output, episode_r], epochs = EPISODE + 1, batch_size = len(episode_output), callbacks = callbacks_list, sample_weight = weights, initial_epoch = EPISODE,verbose=0)

	episode_r = []
	episode_output = []
	episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
	episode_critic = []

	f = open("rewards.txt","a")
	f.write("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Loss: " + str(history.history['loss']) + "\n")
	f.close()

	if EPISODE % 50 == 0: 
		#pygame.event.get()  #讓遊戲畫面能夠更新
		model.save("saved_models/model_updates_" +	str(EPISODE)) 
	EPISODE += 1


