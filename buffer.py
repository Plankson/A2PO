import numpy as np
import torch
import tqdm
import d4rl
import gym
import h5py

def get_env_dataset(env_name, finetune='None', dataset_path=None):
	if env_name.find("random-medium") != -1 or env_name.find('random-expert') != -1:
		sim_env_name = f"{env_name.split('-')[0]}-expert-v2"
		env = gym.make(sim_env_name)
		data = h5py.File(f"{dataset_path}/{env_name}_spec.hdf5", "r")
		dataset = dict(
			observations=[],
			next_observations=[],
			actions=[],
			rewards=[],
			terminals=[],
			timeouts=[],
			init_actions=[],
			returns=[]
		)
		keys = ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
		for k in keys:
			dataset[k] = data[k][:]
	else:
		sim_env_name=env_name
		env = gym.make(sim_env_name)
		dataset = d4rl.qlearning_dataset(env)
	if finetune!='lapo_tune':
		dataset['rewards'] *=100.0
	elif finetune=='cql_tune':
		dataset['rewards']  = (dataset['rewards'] -0.5 ) * 4.0
	return env, dataset

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(2e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.returns = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def sample_with_return(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			torch.FloatTensor(self.returns[ind]).to(self.device)
		)

	def process_sample(self, step, batch_size):
		bb = int(self.trajectory_cnt/10)
		id = int(step/10) + 1
		tra_ind = np.random.randint(0, min((id+1)*bb, self.trajectory_cnt), size=batch_size)
		p_ind = np.array([ np.random.randint(0, self.trajectory_len[tra], size=1) for tra in tra_ind])
		p_ind = p_ind.reshape(p_ind.shape[0])
		return (
			torch.FloatTensor(self.trajectory_s[tra_ind, p_ind]).to(self.device),
			torch.FloatTensor(self.trajectory_a[tra_ind, p_ind]).to(self.device),
			torch.FloatTensor(self.trajectory_nexts[tra_ind, p_ind]).to(self.device),
			torch.FloatTensor(self.trajectory_r[tra_ind, p_ind]).to(self.device),
			torch.FloatTensor(self.trajectory_not_done[tra_ind, p_ind]).to(self.device)
		)

	def transform(self, list_np):
		max_len = np.max([data.shape[0] for data in list_np])
		dim = list_np[0].shape[1]
		new_array = np.zeros((len(list_np), max_len, dim))
		for index, data in enumerate(list_np):
			new_array[index][:len(data)] = data
		return new_array

	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.next_action = self.action[:, :]
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]-1
		print("Load D4RL dataset finished!")

	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std

	def get_return(self):
		pre_return=0
		for i in reversed(range(self.size)):
			self.returns[i] = self.reward[i] + 0.99 * pre_return * self.not_done[i]
			pre_return = self.returns[i]
