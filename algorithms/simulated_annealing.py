import numpy as np
import copy
import time
import tqdm
import torch

from common import PolicyNetwork, DiscretePolicyNetwork, reward_fcn
from utils import setup_model_saving

def simulated_annealing_alg(env, *,
                            param_min = -1.,
                            param_max = 1.,
                            num_episodes_avg = 10 ,
                            max_iter = 100, 
                            NNparams_0 = None,
                            initial_temp = 1e5, 
                            max_time = 30, # maximum execution time in seconds
                            ):
    
    assert initial_temp > 0 , "inital_temp must be a positive value"
    
    # Define path to store best policies
    save_path = setup_model_saving(algorithm="SA")

    # Initialize buffers to store data for plotting
    reward_history = []
    std_history = []
    best_reward_history = []

    # Policy initialization
    if env.name == "InventoryManagement":
        policy_net = PolicyNetwork(input_size=env.observation_space.shape[0], 
                                output_size=env.action_space.shape[0],
                                )
    elif env.name == "DiscreteInventoryManagement":
        policy_net = DiscretePolicyNetwork(input_size=env.observation_space.shape[0], 
                                           output_size=env.action_space.n,
                                           )

    # # INITIALIZATION
    # Parameters
    current_param = policy_net.state_dict() if NNparams_0 is None else NNparams_0 
    best_param = copy.deepcopy(current_param)
    # Rewards
    current_reward , std = reward_fcn(policy_net, env, num_episodes=num_episodes_avg)
    best_reward = copy.deepcopy(current_reward)
    reward_history.append(best_reward)
    std_history.append(std)
    best_reward_history.append(best_reward)
    # Iteration counter
    i = 0
    # Timer
    start_time = time.time()

    # OPTIMIZATION LOOP
    for i in tqdm.tqdm(range(max_iter)):
               
        # Sample a new policy from randomly
        candidate_param = sample_params(current_param, param_min, param_max)

        # Evaluate the candidate policy
        policy_net.load_state_dict(candidate_param)
        candidate_reward, std = reward_fcn(policy_net, env, num_episodes=num_episodes_avg)

        # Check if the candidate policy is better than the current one
        if candidate_reward > best_reward:
            # Update the new best policy
            best_reward = candidate_reward
            best_param = copy.deepcopy(candidate_param)
            # Save policy
            torch.save(best_param, save_path)
            
        # Check if the candidate policy should be kept or discarded
        diff = candidate_reward - current_reward
        temp = initial_temp / (1 + i) # update temperature paramter
        metropolis = np.exp(diff/temp) # compute metropolis acceptance probability
        if diff > 0 or np.random.rand() < metropolis:
            # Update the current policy 
            current_param = copy.deepcopy(candidate_param)
            current_reward = candidate_reward
        
        # Store the data for plotting
        reward_history.append(candidate_reward)
        std_history.append(std)
        best_reward_history.append(best_reward)

        # Check execution time
        if (time.time() - start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    # Pack data for plotting
    plot_data = {'reward_history': reward_history,
                'std_history': std_history,
                'best_reward_history': best_reward_history}

    print(f"Policy model weights saved in: {save_path}")
    print(f"Best reward found during training: {best_reward}")
            
    return save_path , plot_data

def sample_params(params_prev, param_min, param_max):
    '''
    Sample a random point in the neighborhood of a given point or value or the parameters (v). Tailored for EXPLOITATION purposes

    Explanation:
    sign = (torch.randint(2, (v.shape)) * 2 - 1) # This returns either -1 or 1
    eps = torch.rand(v.shape) * (param_max - param_min) # This returns the width of the step to be taken in the modification of the parameters
    Hence, the total update is: v + sign*eps.
    '''
    params = {k: torch.rand(v.shape) * (param_max - param_min) * (torch.randint(2, (v.shape))*2 - 1) + v \
              for k, v in params_prev.items()}
    return params