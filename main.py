import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import datetime
import buffer
from buffer import ReplayBuffer
from a2po import OffRL
from tensorboardX import SummaryWriter
from datetime import datetime
import h5py
from tqdm import tqdm
now = datetime.now()


current_time = now.strftime("%H:%M:%S")
def eval_policy(policy, env, seed, mean, std, seed_offset=100, eval_episodes=5):
	env.seed(seed + seed_offset)
	ep_ret_list = []
	for _ in range(eval_episodes):
		state, done = env.reset(), False
		epi_reward=0.
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = env.step(action)
			epi_reward += reward
		print( f'{epi_reward}', end=", ")
		ep_ret_list.append(env.get_normalized_score(epi_reward) * 100)

	avg_reward =np.mean(ep_ret_list)
	return avg_reward


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--algo", default="AC")        				# OpenAI gym environment name
	parser.add_argument("--env", default="halfcheetah-expert-v2")   # OpenAI gym environment name
	parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=4e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	# TD3
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)    # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

	parser.add_argument("--use_discrete", default=False, type=bool) # whether use discrete \xi
	parser.add_argument("--use_epsilon", default=False, type=bool) # whether use discrete \xi
	parser.add_argument("--epsilon", default=-0.0, type=float)      # positive threshold \epsilon

	parser.add_argument("--bc_weight", default=1.0, type=float)     # BC term weight
	parser.add_argument("--use_cuda", default=True, type=bool)      # whether use gpu
	parser.add_argument("--vae_step", default=200000, type=int)     # VAE train step K
	parser.add_argument("--alpha", default=1.0, type=float)       	# max Q weight
	parser.add_argument("--normalize", default=True)       			# Q-normalization
	parser.add_argument("--reward_tune", default='None')   # finetune reward
	parser.add_argument("--clip", default=False, type=bool)            # finetune reward
	parser.add_argument("--in_sample", default=False, type=bool)

	args = parser.parse_args()
	model_path = f"./{args.env}/"
	data_path = f"./{args.env}/"
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	if args.use_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("---------------------------------------")
	print(f"Setting: Training {args.algo}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")
	env, dataset = buffer.get_env_dataset(args.env, args.reward_tune)
	if args.env.find('medium-diverse') !=-1 or args.env.find('large-diverse')!=-1:
		min_v = -100.0
		max_v = 100.0
	else:
		min_v = dataset['rewards'].min()/(1-args.discount)
		max_v = dataset['rewards'].max()/(1-args.discount)
	l_thresold, r_thresold = -args.epsilon, args.epsilon
	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"device": device,
		"min_v": min_v,
		"max_v": max_v,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"alpha": args.alpha,
		# AC
		# "epsilon": args.epsilon,
		"l_thresold": l_thresold,
		"r_thresold": r_thresold,
		"bc_weight": args.bc_weight,
		"vae_step": args.vae_step,
		"use_discrete": args.use_discrete,
	}

	# Initialize policy
	policy = OffRL(**kwargs)

	replay_buffer = ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(dataset)
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	writer = SummaryWriter(
		logdir=f'./runs/{args.algo}_{args.env}_{args.seed}_{current_time}'
	)
	evaluations = []
	for t in tqdm(range(int(args.max_timesteps)), desc='PI training', ncols=75):
		policy.policy_train(replay_buffer,writer, args.algo, args.env, args.clip, args.batch_size)
		# Evaluate episode
		if t % args.eval_freq == 0:
			eval_res = eval_policy(policy, env, args.seed, mean, std)
			evaluations.append(eval_res)
			# writer.add_scalar(f'{args.env}/eval_reward', eval_res, t)
			print(f"| {args.algo} | {args.env}_{args.seed} | iterations: {t} | eval_reward: {eval_res} |")
			evaluations.append(eval_res)
			writer.add_scalar(f'{env}/final_return', eval_res, t)
			np.save(f"{data_path}{args.seed}.npy", evaluations)

