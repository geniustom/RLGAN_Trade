import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


ENV_NAME = 'Pendulum-v0' #'Pendulum-v0' 'CartPole-v0' 'BreakoutDeterministic-v4'


# Get the environment and extract the number of actions.
def CreateENV():
	env = gym.make(ENV_NAME)
	nb_actions = env.action_space.shape[0]
	
	print(nb_actions,env.action_space.shape)
	return env,nb_actions


# Next, we build a very simple model.
def BuildDDPGModel(env,nb_actions):
	actor = Sequential()
	actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
	actor.add(Dense(16))
	actor.add(Activation('relu'))
	actor.add(Dense(16))
	actor.add(Activation('relu'))
	actor.add(Dense(16))
	actor.add(Activation('relu'))
	actor.add(Dense(nb_actions))
	actor.add(Activation('linear'))
	#print(actor.summary())

	action_input = Input(shape=(nb_actions,), name='action_input')
	observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
	flattened_observation = Flatten()(observation_input)
	x = Concatenate()([action_input, flattened_observation])
	x = Dense(32)(x)
	x = Activation('relu')(x)
	x = Dense(32)(x)
	x = Activation('relu')(x)
	x = Dense(32)(x)
	x = Activation('relu')(x)
	x = Dense(nb_actions)(x)
	x = Activation('linear')(x)
	critic = Model(inputs=[action_input, observation_input], outputs=x)
	#print(critic.summary())
	
	return actor,critic,action_input


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
def CreateAgent(actor,critic,action_input,nb_actions,weight_path=None):
	memory = SequentialMemory(limit=100000, window_length=1)
	random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
	print(nb_actions,action_input,memory,random_process)
	agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
	                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
	                  random_process=random_process, gamma=.99, target_model_update=1e-3)
	agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

	try:	agent.load_weights(weight_path)
	except: print("No model load,train new model")
	return agent
	


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using Ctrl + C.
def Train(agent,env,nb_steps=50000,weight_path=None):
	agent.fit(env, nb_steps, visualize=True, verbose=1, nb_max_episode_steps=200)
	if weight_path!=None:	agent.save_weights(weight_path, overwrite=True)


def main():
	env,nb_actions=CreateENV()
	actor,critic,action_input=BuildDDPGModel(env,nb_actions)
	
	# After training is done, we save the final weights.
	path='ddpg_{}.h5f'.format(ENV_NAME)
	agent=CreateAgent(actor,critic,action_input,nb_actions,weight_path=path)
	Train(agent,env,nb_steps=10000,weight_path=path)
	

	# Finally, evaluate our algorithm for 5 episodes.
	#agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)	
	

if __name__ == "__main__":
	main()