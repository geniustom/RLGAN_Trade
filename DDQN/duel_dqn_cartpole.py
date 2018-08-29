import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from simple_env import CartPoleEnv

ENV_NAME = 'Test'


# Get the environment and extract the number of actions.
def CreateENV():
	#env = gym.make(ENV_NAME)
	#nb_actions = env.action_space.n
	
	env = CartPoleEnv()
	nb_actions = env.action_space.n
	
	print(nb_actions,env.observation_space.shape)
	return env,nb_actions



# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
def BuildDDQNModel(env,nb_actions):
	model = Sequential()
	model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(nb_actions))
	model.add(Activation('linear'))
	#print(model.summary())
	
	return model

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
def CreateAgent(model,nb_actions,weight_path=None):
	memory = SequentialMemory(limit=100000, window_length=1)
	policy = BoltzmannQPolicy()
	# enable the dueling network , you can specify the dueling_type to one of {'avg','max','naive'}
	agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=0,
				   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
	agent.compile(Adam(lr=1e-3), metrics=['mae'])

	try: agent.load_weights(weight_path)
	except: print("No model load,train new model")
	return agent


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using Ctrl + C.
def Train(agent,env,nb_steps=50000,weight_path=None):
	agent.fit(env, nb_steps=nb_steps, visualize=True, verbose=1)
	if weight_path!=None:   agent.save_weights(weight_path, overwrite=True)



def main():
	env,nb_actions=CreateENV()
	model=BuildDDQNModel(env,nb_actions)
	
	# After training is done, we save the final weights.
	path='ddqn_{}.h5f'.format(ENV_NAME)
	agent=CreateAgent(model,nb_actions,weight_path=path)
	Train(agent,env,nb_steps=100000,weight_path=path)
	

	# Finally, evaluate our algorithm for 5 episodes.
	agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)    
	

if __name__ == "__main__":
	main()