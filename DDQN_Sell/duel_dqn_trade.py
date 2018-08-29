import numpy as np

from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten,Dropout,Conv1D,Conv2D, Input,Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from trade_env import TradeEnv

ENV_NAME = 'Test'


# Get the environment and extract the number of actions.
def CreateENV(mode='train'):
	env = TradeEnv(mode=mode)
	nb_actions = env.action_space.n
	
	print("actions:",nb_actions,",input shape:",env.observation_space.shape)
	return env,nb_actions



# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
def BuildDDQNModel(env,nb_actions):
	#shape=(1,env.observation_space.shape[1]*env.observation_space.shape[0])
	model = Sequential()
	model.add(Reshape((8, 300), input_shape=(1,) +env.observation_space.shape))
	model.add(Conv1D(128, (2), strides=1, padding='valid', activation='relu'))
	model.add(Conv1D(128, (2), strides=1, padding='valid', activation='relu'))
	model.add(Conv1D(128, (2), strides=1, padding='valid', activation='relu'))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(256))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))

	print(model.summary())

	return model

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
def CreateAgent(model,nb_actions,weight_path=None):
	memory = SequentialMemory(limit=500000, window_length=1)
	policy = BoltzmannQPolicy()
	# enable the dueling network , you can specify the dueling_type to one of {'avg','max','naive'}
	agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1,
				   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
	agent.compile(Adam(lr=1e-3), metrics=['mae'])

	try: agent.load_weights(weight_path)
	except: print("No model load,train new model")
	return agent


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using Ctrl + C.
def Train(agent,env,nb_steps,weight_path=None):
#	for i in range(0, nb_steps, 10000):
	agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1)
    # After training is done, we save the final weights.
	if weight_path!=None:   agent.save_weights(weight_path, overwrite=True)



def main():
	path='ddqn_{}.h5f'.format(ENV_NAME)
    
#	# Initial, training
#	env,nb_actions=CreateENV()
#	model=BuildDDQNModel(env,nb_actions)
#	agent=CreateAgent(model,nb_actions,weight_path=path)
#	Train(agent,env,nb_steps=1000000,weight_path=path)
	

	# Finally, evaluate
	env,nb_actions=CreateENV(mode='test')
	model=BuildDDQNModel(env,nb_actions)
	agent=CreateAgent(model,nb_actions,weight_path=path)
	agent.test(env, nb_episodes=150, visualize=False, nb_max_episode_steps=200)    
	

if __name__ == "__main__":
	main()
    
    
    
    
    
    
    
    
    
    
    
'''
def BuildDDQNModel(env,nb_actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	model.add(Dense(2400))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1200))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(600))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))
	print(model.summary())
	
	return model
'''