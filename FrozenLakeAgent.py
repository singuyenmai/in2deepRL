"""
Example:
from FrozenLakeAgent import Agent
agent = Agent()

# check initial q-table
agent.q_table

agent.play()

# train agent
agent.train(10000)

# recheck q-table
agent.q_table

# play agent
agent.play()
"""

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

# Initializing Q-learning parameters
num_episodes = 30000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

class Agent():
	def __init__(self):
		# Create the environment
		self.env = gym.make("FrozenLake-v0")

		# Create the Q-table
		action_space_size = self.env.action_space.n
		state_space_size = self.env.observation_space.n

		# Outputs
		#self.q_table = np.zeros((state_space_size, action_space_size)) # TODO: better random initialization?
		self.q_table = np.random.rand(state_space_size, action_space_size)*0.001
		self.rewards_all_episodes = []

	def train(self, num_episodes=num_episodes):
		""" Implementation of Q-learning algorithm
		"""
		global max_steps_per_episode
		global learning_rate
		global discount_rate
		global exploration_rate
		global max_steps_per_episode
		global min_exploration_rate
		global exploration_decay_rate

		# Training
		print(f'TRAINING IN {num_episodes} EPISODES')
		for episode in range(num_episodes):
			# initialize new episode params
			state = self.env.reset()
			done = False
			reward_current_episode = 0

			for step in range(max_steps_per_episode):
				# Exploration-exploitation trade-off
				exploration_rate_threshold = random.uniform(0, 1)
				if exploration_rate_threshold > exploration_rate:
					action = np.argmax(self.q_table[state, :])
				else:
					action = self.env.action_space.sample()

				# Take new actions
				new_state, reward, done, info = self.env.step(action)

				# Update Q-table for Q(s, a)
				self.q_table[state, action] = self.q_table[state, action] * (1 - learning_rate) + \
										learning_rate * (reward + discount_rate * np.max(self.q_table[new_state, :]))
				# Set new state
				state = new_state
				# Add new reward
				reward_current_episode += reward	

				if done == True:
					break

			if (episode+1)%1000 == 0:
				_reward = sum(self.rewards_all_episodes[-1000:])/1000
				print('Episode: {}/{} | reward: {:.5f}'.format(episode+1, num_episodes, _reward))

			# Exploration rate decay
			exploration_rate = min_exploration_rate + \
							(max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
			# Add new episode reward to total reward list
			self.rewards_all_episodes.append(reward_current_episode)


		# Print updated Q-table
		print("\n\n********Q-table********\n")
		print(self.q_table)
		print('Training done')

	def play(self, num_replay=3):
		"""Watch out agent play Frozen Lake by playing the best action 
		according to the Q-table
		"""
		global max_steps_per_episode
		for episode in range(num_replay):
			# initialize new episode params 
			state = self.env.reset()
			done = False
			print("****** EPISODE ", episode+1, "**** \n")
			time.sleep(1)

			for step in range(max_steps_per_episode):
				# Show current state of env on screen
				clear_output(wait=True)
				self.env.render()
				time.sleep(0.3)

				# Choose action with highest Q-value for current state
				action = np.argmax(self.q_table[state, :])

				# Take new action
				new_state, reward, done, info = self.env.step(action)

				if done:
					clear_output(wait=True)
					self.env.render()
					if reward == 1:
						# Agent reached the goal and won episode
						print("**** You reached the goal! ****")
						time.sleep(2)
					else:
						# Agent stepped in a hole and lost episode
						print("**** You fell through a hole! ****")
						time.sleep(2)
						clear_output(wait=True)
					break
				# Set new state
				state = new_state

		self.env.close()


if __name__ == '__main__':
	agent = Agent()
	print('**** Initial q-table **** ')
	print(agent.q_table)
	agent.train()
 