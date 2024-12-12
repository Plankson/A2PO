import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import math

class Actor(nn.Module):
    def __init__(self, state_dim, latent_dim, max_action, device):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, latent_dim)
        self.device = device
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_latent(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return (self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)
        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, 256)
        self.l6 = nn.Linear(256, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 1)
        # V architecture
        self.v1 = nn.Linear(state_dim, 256)
        self.v2 = nn.Linear(256, 256)
        self.v3 = nn.Linear(256, 256)
        self.v4 = nn.Linear(256, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))
        q2 = F.relu(self.l5(torch.cat([state, action], 1)))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def get_latent_z(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        return q1

    def Q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))
        return q1

    def v(self, state):
        v = F.relu(self.v1(state))
        v = F.relu(self.v2(v))
        v = F.relu(self.v3(v))
        v = (self.v4(v))
        return v

# Vanilla Variational Auto-Encoder
class CVAE(nn.Module):
    def __init__(self, input_dim, l_dim, output_dim, latent_dim, max_action, device):
        super(CVAE, self).__init__()
        self.e1 = nn.Linear(input_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        self.d1 = nn.Linear(l_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, output_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, input, latent_input):
        z = F.relu(self.e1(input))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.FloatTensor(np.random.normal(0, 1, size=(std.size()))).to(self.device)
        u = self.decode(latent_input, z)
        return u, mean, std

    def decode_softplus(self, latent_input, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(latent_input.size(0), self.latent_dim))).to(
                self.device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([latent_input, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode(self, latent_input, z=None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(latent_input.size(0), self.latent_dim))).to(
                self.device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([latent_input, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

    def decode_multiple(self, latent_input, z=None, num_decode=10):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0, 1, size=(latent_input.size(0), num_decode, self.latent_dim))).to(
                self.device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([latent_input, z], 2)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))


class OffRL(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            min_v,
            max_v,
            device,
            tau=0.005,
            discount=0.99,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            l_thresold=0.0,
            r_thresold=0.0,
            bc_weight=1.0,
		    vae_step=200000,
            use_discrete=False,
            doubleq_min=1.0
    ):
        latent_dim = action_dim * 2
        adv_dim = 1
        self.device = device
        self.actor = Actor(state_dim+1 , latent_dim, max_action, device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)
        self.vae = CVAE(state_dim+adv_dim+action_dim, state_dim+adv_dim, action_dim, action_dim*2, max_action, self.device).to(device)
        self.vae_target = copy.deepcopy(self.vae)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=3e-4)
        self.vae_scheduler = torch.optim.lr_scheduler.StepLR(self.vae_optimizer, step_size=1000, gamma=0.9)
        with torch.no_grad():
            self.adv_dist = Normal(torch.tensor(0.5).to(device), torch.tensor(1.0).to(device))
            self.adv_clamp=[0.0,1.0]
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.log_freq = 1000
        self.bc_weight=bc_weight
        self.total_it = 0
        self.l_thresold=l_thresold
        self.r_thresold=r_thresold
        self.vae_step=vae_step
        self.use_discrete = use_discrete
        self.doubleq_min=doubleq_min
        self.min_v = min_v
        self.max_v = max_v

    def select_action(self, state, level=1.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        adv = torch.full((state.shape[0], 1), level).to(self.device)
        state = torch.concat((state, adv), dim=1)
        latent_a = self.actor(state)
        a = self.vae.decode(state, latent_a).cpu().data.numpy().flatten()
        return a

    def select_action_latent(self, state, level=1.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        adv = torch.full((state.shape[0],1), level).to(self.device)
        state = torch.concat((state, adv), dim=1)
        latent_a = self.actor.get_latent(state)
        return latent_a
    def cvae_action(self, state, level=1.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        adv = torch.full((state.shape[0],1), level).to(self.device)
        state = torch.concat((state, adv), dim=1)
        a = self.vae.decode(state).cpu().data.numpy().flatten()
        return a

    def TD3BC_critic_loss(self, s0, a0, r0, nd0, s1, clip=False):
        with torch.no_grad():
            noise = (
                    torch.randn_like(a0) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            adv_sample = torch.ones((s1.size()[0], 1)).to(self.device)
            latent_a = self.actor_target(torch.cat((s1, adv_sample), dim=1))
            hat_a1 = (
                    self.vae.decode(torch.cat((s1, adv_sample), dim=1), latent_a) + noise
            ).clamp(-self.max_action, self.max_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(s1, hat_a1)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r0 + nd0 * self.discount * target_Q
            if clip:
                target_Q = target_Q.clamp(self.min_v, self.max_v)
            target_V = target_Q
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(s0, a0)
        current_V = self.critic.v(s0)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + F.mse_loss(current_V, target_V)
        return critic_loss

    def TD3BC_actor_loss(self, s):
        latent_a = self.actor(s)
        pi = self.vae.decode(s, latent_a)
        Q = self.critic.Q1(s, pi)
        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean()
        return actor_loss

    def cvae_loss(self, x, latent_x, label, cvae_model, weight=None):
        rec, mean, std = cvae_model(x, latent_x)
        rec_loss = torch.pow((rec-label),2).mean(dim=1)
        kl_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        vae_loss = rec_loss + 1.0 * kl_loss
        if weight != None:
            vae_loss = vae_loss * weight
        vae_loss = vae_loss.mean()
        return vae_loss

    def reg_loss(self, hat_a, a,weight=None):
        reg_loss = torch.pow(a-hat_a, 2)
        if weight!=None:
            reg_loss = reg_loss * weight
        reg_loss = reg_loss.mean()
        return reg_loss

    def policy_train(self, replay_buffer, logger, algo, env, clip=False, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        not_done = not_done.to(self.device)

        critic_loss = self.TD3BC_critic_loss(state, action, reward, not_done, next_state, clip)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.total_it <= self.vae_step:
            adv = self.critic.Q1(state, action) - self.critic.v(state)
            adv =torch.tanh(adv)
            if self.use_discrete:
                adv = torch.where(adv< self.l_thresold, torch.tensor(-1, dtype=adv.dtype).to(self.device),
                                  torch.where(adv > self.r_thresold,
                                              torch.tensor(1, dtype=adv.dtype).to(self.device),
                                              torch.tensor(0, dtype=adv.dtype).to(self.device)))

            adv = adv.detach()
            vae_loss = self.cvae_loss(torch.cat((state, adv, action), dim=1), torch.cat((state, adv), dim=1), action, self.vae)
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            adv_sample = torch.ones((state.size()[0], 1)).to(self.device)
            latent_best_pi = self.actor(torch.cat((state, adv_sample),dim=1))
            best_pi = self.vae.decode(torch.cat((state, adv_sample), dim=1), latent_best_pi)
            Q_a = self.critic.Q1(state, best_pi)
            lmbda = self.alpha / Q_a.abs().mean().detach()
            batch_adv = self.critic.Q1(state, action)-self.critic.v(state)
            batch_adv = torch.tanh(batch_adv)
            if self.use_discrete:
                batch_adv = torch.where(batch_adv< self.l_thresold, torch.tensor(-1, dtype=batch_adv.dtype).to(self.device),
                                        torch.where(batch_adv > self.r_thresold,
                                                    torch.tensor(1, dtype=batch_adv.dtype).to(self.device),
                                                    torch.tensor(0, dtype=batch_adv.dtype).to(self.device)))
            batch_adv = batch_adv.detach()
            latent_pi = self.actor(torch.cat((state, batch_adv), dim=1))
            pi = self.vae.decode(torch.cat((state, batch_adv), dim=1), latent_pi)
            actor_loss = -lmbda * Q_a.mean() + self.bc_weight * self.reg_loss(pi, action)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % self.log_freq == 0:
            if self.total_it <= self.vae_step:
                logger.add_scalar(f'{env}/vae_loss', vae_loss.item(), self.total_it)
            logger.add_scalar(f'{env}/critic_loss', critic_loss.item(), self.total_it)
            logger.add_scalar(f'{env}/actor_loss', actor_loss.item(), self.total_it)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.vae.state_dict(), filename + "_vae")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename+"_critic"))
        self.actor.load_state_dict(torch.load(filename+"_actor"))
        self.vae.load_state_dict(torch.load(filename+"_vae"))
