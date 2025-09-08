import numpy as np 
# from sklearn.neural_network import MLPRegressor # not used because loss fuction cannot be customized in sklearn
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal 

import os
import time
from utils import setup_logging, setup_model_saving
'''
REFERENCES:
# https://github.com/OptiMaL-PSE-Lab/REINFORCE-PSE
# train nn from scratch using scipy minimize https://euanrussano.github.io/20190821neuralNetworkScratch/
# https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
'''

class RolloutBuffer():
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        # del list[idx_to_delete] is the syntax used to empty the lists 
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions, action_std_init=.5):
        super(ActorCritic, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.action_var = torch.full((self.num_actions,), action_std_init * action_std_init)

        torch.manual_seed(24)
        self.actor = nn.Sequential(
                        nn.Linear(self.num_states, 256),
                        nn.Tanh(),
                        nn.Linear(256, 256),
                        nn.Tanh(),
                        nn.Linear(256, self.num_actions),
                        nn.ReLU() # TODO: check if this activation is appropriate for the action space
                        )

        self.critic =  nn.Sequential(
                        nn.Linear(self.num_states, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),
                        )   

    def set_action_std(self, action_std):
        self.action_var = torch.full((self.num_actions,), action_std * action_std)

    def act(self, state):
        '''
        Method to sample an action a_t given state s_t using the policy learnt by the Actor

        Returs:
        - action : Chosen action a_t ~ actor(s_t)
        - action_logprob : log probability (or likelyhood) of taking a_t
        '''
        action_mean = self.actor(state) # Returns a tensor where each element is the mean of the action distribution. The sum of all the elements in this tensor adds up to 1.
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # ASSUMPTION: independent actions, so the covariance matrix is diagonal with variance = action_variance
        dist = MultivariateNormal(action_mean, cov_mat) # Creates a multivariate normal distribution

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()
         
    def evaluate(self, state, action):
        '''
        Method to evaluate policy given by the actor network. 

        Returns:
        - action_logprob : log probability of the action taken >>> Used to compute importance sampling weights
        - state_values : V(s_t) or expected return from state s_t >>> Used to compute the advantage function Adv(s_t, a_t) = Q(s_t, a_t) - V(s_t) = r_t + gamma * V(s_{t+1}) - V(s_t)
        - entropy : Entropy of the action distribution >>> Used to encourage exploration by discouraging deterministic policies
        '''

        # STEP 1: Generate a distribution from the current prediction of the actor network
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        # STEP 2: Evaluate the log probability of the action of interest in under the current policy 
        action_logprob = dist.log_prob(action)

        # STEP 3: Evaluate policy by computing the value of the current state
        value_state = self.critic(state)

        return action_logprob, value_state , dist.entropy()     

    def forward(self):
        raise NotImplementedError 

class PPO:
    def __init__(self, num_states, num_actions, lr_actor=5e-4, lr_critic=1e-3, action_std_init = 0.5, action_std_min = 0.05, decay_rate=.03, decay_freq=1e4, num_epochs = 100 , gamma = 1, eps_clip = .2, max_train_steps = 364*10000, update_timestep = 364/2, log_freq = 1):
        
        self.action_std = action_std_init
        self.action_std_min = action_std_min
        self.decay_rate = decay_rate 
        self.decay_freq = decay_freq 

        self.policy = ActorCritic(num_states, num_actions ) # Policy to be optimized
        self.policy_old = ActorCritic(num_states, num_actions ) # Policy to sample trajectories from

        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        
        self.buffer = RolloutBuffer() # Buffer to store trajectories

        # PPO Hyperparameters
        self.update_timestep = int(update_timestep) # Number of timesteps to collect before updating the policy
        self.num_epochs = num_epochs # Number of epochs/policy updates per batch of data
        self.gamma = gamma # Adv(s_t,a_t) = Q(s_t,a_t) - V(s_t) = r_t + gamma*V(s_{t+1}) - V(s_t)
        self.eps_clip = eps_clip # Value of the clipping parameter in the PPO loss function
        self.max_train_steps = int(max_train_steps) # break training loop if timeteps > max_training_timesteps

        # Set up logging 
        self.log_dir = os.getcwd() + '/PPO_logs' # Directory where the logs will be saved
        self.log_freq = log_freq # Save logs every log_freq episodes
        self.log_f_name = None # Name of the log file to be created

        # Set up path to save the policy
        self.save_path = os.getcwd() + '/PPO_policy.pt' # Path to save the policy

        # Flag to indicate if running on test mode
        self.test_mode = False
        self.save_freq = 1 # Number of episodes to wait before saving a copy of the policy

    def select_action(self, state):
        '''
        Method to select an action given a state using the policy network
        '''
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        action.round_(decimals=0) # NOTE: action describes number of items of product ordered. Hence, round action to nearest integer to get discrete numbers

        # Save (state, action, logprob) to buffer
        self.buffer.states.append(state) 
        self.buffer.actions.append(action) 
        self.buffer.logprobs.append(action_logprob) 

        return action.detach().numpy().flatten() 

    def update(self):

        ### Compute reward-to-go (meaning return = cumulative discounted reward) for each (state, action) pair in the buffer (Step 4 in OpenAI's PPO paper)
        discounted_reward = 0
        rewards = []
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0,discounted_reward) # insert method is used to fill a list from the top (opposite to append). This is done by specifying pos=0, given that the syntax is insert(pos,elmnt)
        rewards = torch.tensor(rewards, dtype=torch.float32) 

        ### Normalize rewards. PPO is very sensitive to the scale of the loss funciton. If rewards are too high or too low, updated can be erratic.
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        ### Convert lists to tensors
        '''
         The RolloutBuffer object contains lists, which must be transformed into tensors before being used.
         PROCEDURE:
         1. stack: converts lists to a single tensor of shape [N, num_states] (if each state or row of the list is another list of shape [num_states]. If each row is a list of shape [1,num_states], the tensor will be [N,1,num_states])
         2. squeeze: removes unnecessary dimensions (e.g., [N,1,num_states] to [N,num_states])
         3. detach: breaks the tensor from the computational graph
         4. (not used) to_device(device): moves the tensor to the specified device (cpu or cuda)
        '''
        batch_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        batch_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        batch_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        ### Update Actor & Critic networks   
        for _ in range(self.num_epochs):
            ## Compute advantage estimate based on current value function (Step 5 in OpenAI's PPO paper)
            action_logprobs, state_values , _ = self.policy.evaluate(batch_states, batch_actions) 
            ratio = torch.exp(action_logprobs - batch_logprobs.detach())         
            advantage = rewards - state_values.squeeze().detach() 
          
            ## Update actor (Step 6 in OpenAI's PPO paper)
            loss_actor = (-1) * torch.min( ratio * advantage ,
                                           torch.clamp(ratio, 1-self.eps_clip , 1+self.eps_clip) * advantage)
            self.optimizer_actor.zero_grad()
            loss_actor.mean().backward()
            self.optimizer_actor.step()

            ## Update critic (Step 7 in OpenAI's PPO paper)
            loss_critic = torch.mean((rewards - state_values)**2) 
            self.optimizer_critic.zero_grad()
            loss_critic.mean().backward() # mean() because we want the MSE
            self.optimizer_critic.step()
        
        ### Update process of "policy" is over after num_epochs >>> Unfreeze "policy_old" and copy weights from "policy" to "policy_old"
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())

        ### Clear buffer
        self.buffer.clear()

    def train(self, env, logging = True, verbose = False, model_dict = dict()):

        ### Set up logging
        if logging:
            self.log_f_name = setup_logging()
            log_f = open(self.log_f_name, 'w+')
            log_f.write("Episode,Timestep,Reward\n")

        ### Initialize counters to track the number of environment time steps and the number of episodes
        count_timesteps = 0
        count_episodes = 0
        count_updates = 0

        log_running_reward = 0 
        log_running_episodes = 0 

        ### REPEAT until the maximum number of training steps is reached
        while count_timesteps < self.max_train_steps:
            
            env.reset() # (Re)Intialize environment
            episode_terminated = False # Variable to track if the current episode has terminated

            current_reward = 0 # Stores total UNDISCOUNTED reward accumulated in the current episode (used for monitoring purposes)
            
            ### REPEAT for every time step inside the current episode 
            while not episode_terminated:
                ## Observe state, sample action, take reward and move to next state
                state = env.state
                action = self.select_action(state)

                ## Save state, action, and logprob to buffer: ALREADY DONE IN select_action METHOD

                ## Apply action to environment and observe next state, reward, and termination status
                state , reward, episode_terminated, _ = env.step(action)
                
                ## Save reward and termination status to buffer
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(episode_terminated)

                ## Update counters
                count_timesteps += 1
                current_reward += reward
                log_running_reward += reward
            
                ## Update Actor & Critic networks
                '''NOTE THAT Actor & Critic policies are not updated at every time step; only when sufficient experience has been collected.
                Specifically, the policy is updated every "self.update_timestep" time steps.
                '''
                if count_timesteps % self.update_timestep == 0:
                    self.update()
                    count_updates += 1

                # Decay action std of ouput action distribution
                if count_timesteps % self.decay_freq == 0:
                    self.action_std = max(self.action_std * (1 - self.decay_rate), self.action_std_min)
                    self.policy.set_action_std(self.action_std)
                    self.policy_old.set_action_std(self.action_std)

            ### Update episode counter
            count_episodes += 1
            log_running_episodes += 1
            if verbose:
                print(f"Episode: {count_episodes}, Time step: {count_timesteps}, Update nr.: {count_updates}, Undiscounted reward: {current_reward:.2f}")

            ## Check if data should be logged
            if logging and count_episodes % self.log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                # Write to log file
                log_f.write('{:.0f},{:.0f},{:.3f}\n'.format(count_episodes, count_timesteps, log_avg_reward))
                log_f.flush()
                # Reset running reward and episodes counters
                log_running_reward = 0
                log_running_episodes = 0

            ## Check if data should be saved
            if self.test_mode and count_episodes % self.save_freq == 0:
                model_dict['actor_model'] = self.policy.actor.state_dict()

        # Close log file
        if logging:
            log_f.close()

        # Save policy
        self.save_path = setup_model_saving()
        torch.save(self.policy.actor.state_dict(), self.save_path)
        print(f"Actor model weights saved in: {self.save_path}")
        if logging:
            print(f"Log file saved in: {self.log_f_name}")

        return self.save_path, self.log_f_name

    def evaluate_policy(self, env , test_demand_dataset):
        '''
        Evaluates a actor policy in the environment.
        '''
        reward_list= []
        for demand in test_demand_dataset:
            # Set the demand dataset for this episode
            env.demand_dataset = demand

            # Reset the environment, flags and counters
            env.reset()
            state = env.state
            episode_terminated = False
            reward_episode = 0

            # Run the episode
            while not episode_terminated:
                action = self.select_action(state)
                state , reward, episode_terminated, _ = env.step(action)
                reward_episode += reward

            # Store the total reward for this episode
            reward_list.append(reward_episode)

        return reward_list








