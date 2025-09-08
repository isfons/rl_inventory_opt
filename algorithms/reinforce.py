import numpy as np 
import torch
import time
import tqdm 

from torch.distributions import MultivariateNormal

from common import PolicyNetwork, DiscretePolicyNetwork
from utils import *
from IMP_CW_env import MESCEnv

def REINFORCE_alg( env, *, 
                    lr_policy_net= 5e-5, 
                    lr_value_net = 1e-5, 
                    discount_factor = .99 , 
                    max_episodes = 1000 , 
                    max_time = 120. ,
                    weight_entropy = 0.001, 
                    action_std_init = .5):
    
    # Create log file
    log_f_path = setup_logging(algorithm = "REINFORCE")
    log_f = open(log_f_path, 'w+')
    log_f.write("Timestep,Reward\n")
    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm = "REINFORCE")

    # Initialize variables
    counter_timesteps = 0
    best_reward = -np.inf
    start_time = time.time()
    
    # Initialize policies and optimizers    
    if env.name == "InventoryManagement":
        policy_net = PolicyNetwork(input_size=env.observation_space.shape[0], 
                                output_size=env.action_space.shape[0],
                                h1_size = h1_size,
                                h2_size = h2_size,
                                )
    elif env.name == "DiscreteInventoryManagement":
        policy_net = DiscretePolicyNetwork(input_size=env.observation_space.shape[0], 
                                           output_size=env.action_space.n,
                                           )
    value_net = ValueNetwork(input_size=env.observation_space.shape[0])
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr_value_net)

    for i in tqdm.tqdm(range(int(max_episodes))):
        
        # Run an episode and collect experience
        trajectory = {}
        trajectory["values"] = []
        trajectory["actions"] = []
        trajectory["logprobs"] = []
        trajectory["rewards"] = []
        trajectory["entropies"] = []

        done = False
        env.reset()
        state = env.state
        
        while not done:
            action , action_logprob , entropy = choose_action(state, policy_net, action_std_init)
            next_state , reward , done , _ = env.step(action.detach().numpy().flatten())

            trajectory["values"].append(value_net(torch.from_numpy(state).float()))
            trajectory["logprobs"].append(action_logprob)
            trajectory["rewards"].append(reward)
            trajectory["entropies"].append(entropy)

            state = next_state
            counter_timesteps += 1
        
        logprobs = torch.stack(trajectory["logprobs"]).squeeze() # shape : (episode_length, )
        entropies = torch.stack(trajectory["entropies"]).squeeze() # shape : (episode_length, )
        values = torch.stack(trajectory["values"]).squeeze() # shape : (episode_length, )

        # Calculate discounted return at every time step
        discounted_return = 0
        returns = []
        for r in reversed(trajectory["rewards"]):
            discounted_return = r + discount_factor * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns , dtype=torch.float32)

        # Compute policy loss
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / advantages.std()
        loss_policy = (-1) * torch.mean(advantages.detach() * logprobs) + weight_entropy * ((-1) * torch.mean(entropies))

        # Compute value loss
        loss_value = torch.nn.functional.mse_loss(values , returns)

        # Update policy network
        optimizer_policy.zero_grad()
        loss_policy.backward()
        optimizer_policy.step()

        # Update Value Network
        optimizer_value.zero_grad()
        loss_value.backward()
        optimizer_value.step()

        # Write episode undiscounted return to log file
        total_return = round(np.mean(sum(trajectory["rewards"])), 4)
        log_f.write('{:.0f},{:.3f}\n'.format(counter_timesteps, total_return))
        log_f.flush() 

        # Save best policy
        if total_return > best_reward:
            best_reward = total_return
        torch.save(policy_net.state_dict(), save_f_path)

        # Check time
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    # Close log file
    log_f.close()
    
    print(f"Log file saved in: {log_f_path}") 
    print(f"Policy model weights saved in: {save_f_path}") 
    print(f"Best reward: {best_reward}")

    return save_f_path , log_f_path

#################################
# Helper functions
#################################
class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def choose_action(state, policy_net, action_std):
    '''
    Sample action in continuous action space modelled with a Multivariate Normal distribution
    '''
    # Predict action mean from Policy Network
    action_mean = policy_net(torch.from_numpy(state).float())

    # Estimate action variance (decaying action std)
    action_var = torch.full(size=(policy_net.fc3.out_features,) , fill_value = action_std**2)
    cov_mat = torch.diag(action_var).unsqueeze(dim=0) 

    # Generate Multivariate Normal distribution with estimated mean and variance
    dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

    # Sample action
    action = dist.sample()

    # Compute logprob and entropy
    logprob = dist.log_prob(action)
    entropy = dist.entropy()

    return action, logprob , entropy