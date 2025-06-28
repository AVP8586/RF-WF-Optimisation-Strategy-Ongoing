import torch as T
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from parameters import FC1_DIMS, FC2_DIMS, GAMMA, ALPHA, GAE_LAMBDA, POLICY_CLIP, BATCH_SIZE, N, N_EPOCHS

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        # np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return T.tensor(np.array(self.states)), T.tensor(np.array(self.actions)), T.tensor(np.array(self.probs)), T.tensor(np.array(self.vals)), \
            T.tensor(np.array(self.rewards)), T.tensor(np.array(self.dones)), batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class CriticNet(nn.Module):
    def __init__(self, input_dims, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS):
        super(CriticNet, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.lstm1 = nn.LSTM(fc1_dims, fc1_dims, batch_first= True)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        # self.lstm2 = nn.LSTM(fc2_dims, fc2_dims, batch_first= True)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, x):
        x = self.fc1(x)
        # h0 = T.zeros(1, x.size(0), self.lstm1.hidden_size).to(x.device)
        # c0 = T.zeros(1, x.size(0), self.lstm1.hidden_size).to(x.device)
        # x, _ = self.lstm1(x, (h0, c0))
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        # h1 = T.zeros(1, x.size(0), self.lstm2.hidden_size).to(x.device)
        # c1 = T.zeros(1, x.size(0), self.lstm2.hidden_size).to(x.device)
        # x, _ = self.lstm2(x, (h1, c1))
        x = self.ln2(x)
        x = self.relu(x)
        return self.fc3(x)
    
class ActorNet(nn.Module):
    def __init__(self, n_actions, input_dims, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS):
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.lstm1 = nn.LSTM(fc1_dims, fc1_dims, batch_first= True)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        # self.lstm2 = nn.LSTM(fc2_dims, fc2_dims, batch_first= True)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.softmax = nn.Softmax(dim= -1)

    def forward(self, x):
        x = self.fc1(x)
        # h0 = T.zeros(1, x.size(0), self.lstm1.hidden_size).to(x.device)
        # c0 = T.zeros(1, x.size(0), self.lstm1.hidden_size).to(x.device)
        # x, _ = self.lstm1(x, (h0, c0))
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        # h1 = T.zeros(1, x.size(0), self.lstm2.hidden_size).to(x.device)
        # c1 = T.zeros(1, x.size(0), self.lstm2.hidden_size).to(x.device)
        # x, _ = self.lstm2(x, (h1, c1))
        x = self.ln2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)
    
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, 
                 fc1_dims= FC1_DIMS, fc2_dims= FC2_DIMS, chkpt_dir='Net/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch__ppo')
        # self.actor = ActorNet(n_actions, input_dims,
                #  fc1_dims= FC1_DIMS, fc2_dims= FC2_DIMS)
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim= -1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)

        return dist
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS,
                 chkpt_dir='Net/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        # self.critic = CriticNet(input_dims, fc1_dims=FC1_DIMS, fc2_dims=FC2_DIMS)
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.Linear(fc2_dims, 1),
            nn.ReLU(),
            nn.Softmax(dim= -1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, input_dims, n_actions, gammma=GAMMA, alpha=ALPHA, gae_lambda=GAE_LAMBDA,
                 policy_clip=POLICY_CLIP, batch_size=BATCH_SIZE, N=N, n_epochs=N_EPOCHS):  #Some of these will be smaller in real environments
        self.gamma = gammma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        self.positions = [0, 1, -1]

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('...saving models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('...loading models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, wlkfrwd, flag):
        state = observation

        action_probs = self.actor(state)
        
        dist = Categorical(action_probs)
        if wlkfrwd and self.positions[T.argmax(action_probs)] != flag:
            action = T.argmax(action_probs)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        value = self.critic(state).to(self.actor.device)
        value = value.item()

        return action.item(), log_prob.item(), value  
    
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            
            values = vals_arr.numpy()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]* (1-int(done_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            advantage = T.where(T.isnan(advantage), T.ones_like(advantage), advantage)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                if T.isnan(advantage[batch]).any():
                    continue
                states = state_arr[batch].to(self.actor.device)  
                old_probs = old_probs_arr[batch].to(self.actor.device)  
                actions = action_arr[batch].to(self.actor.device) 

                # nan check before input to network
                states = T.where(T.isnan(states), T.ones_like(states)*1e-6, states)
                new_action_probs = self.actor(states)
                dist = Categorical(new_action_probs)
                new_probs = dist.log_prob(actions)

                # nan check before input to network
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                prob_ratio = new_probs.exp() / (old_probs.exp() + 1e-5)
                weighted_probs = advantage[batch]*prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
        self.memory.clear_memory()