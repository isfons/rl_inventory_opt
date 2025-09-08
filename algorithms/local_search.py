import numpy as np
import time
import copy
import tqdm
import torch

from utils import setup_model_saving
from common import PolicyNetwork, DiscretePolicyNetwork, reward_fcn

def local_search_alg(env, *,                    
                     param_min = -1.,
                     param_max = 1.,
                     num_episodes_avg = 10 ,
                     max_iter = 100, 
                     max_time = 30, # maximum execution time in seconds
                     ratio_rs_ls = 0.1, 
                     NNparams_0 = None,
                     radius = 0.3, 
                     shrink_ratio = 0.95,
                     evals_shrink = 10,
                     ):
    
    # Check inputs
    assert radius > 0. , "Initial radius must be a positive value (radius > 0)"
    assert param_min < param_max, "Bounds inconsistency: upper bound must be greater than lower bound"
    assert 0. < shrink_ratio < 1., "Shrink ration must be between 0 and 1"
    assert evals_shrink > 0, "Argument evals_shrink must be a positive number"

    # Define path to store best policies
    save_path = setup_model_saving(algorithm="Your algorithm")

    # Create buffers to store data for plotting purposes
    reward_history = []
    std_history = []
    best_reward_history = []
    radius_list = []

    # Build neural network model
    if env.name == "InventoryManagement":
        policy_net = PolicyNetwork(input_size=env.observation_space.shape[0], 
                                output_size=env.action_space.shape[0],
                                )
    elif env.name == "DiscreteInventoryManagement":
        policy_net = DiscretePolicyNetwork(input_size=env.observation_space.shape[0], 
                                           output_size=env.action_space.n,
                                           )

    # Setup iterations dedicated to random and local search
    iter_rs = round(max_iter * ratio_rs_ls) 

    # ######################################################################################
    # DO NOT MODIFY CODE ABOVE THIS LINE
    # ######################################################################################

    # STEP 1: INITIALIZATION
    # 1A. Initialize counters
    i = 0
    fail_i = 0
    # 1B. Initialize policy parameters
    param = policy_net.state_dict() if NNparams_0 is None else NNparams_0
    best_param = copy.deepcopy(param)

    best_reward, best_std = reward_fcn(policy_net, env, num_episodes=num_episodes_avg)
    reward_history.append(best_reward)
    std_history.append(best_std)
    best_reward_history.append(best_reward)

    # 1C. Initialize timer
    start_time = time.time()

    # OPTIMIZATION LOOP: POLICY SEARCH
    for i in tqdm.tqdm(range(max_iter)):
    # while (i < max_iter) and ((time.time()-start_time) < max_time):
        # STEP 2: sample parameters
        if i <= iter_rs:
            # Sample a random policy
            param = sample_uniform_params(param, param_min, param_max)
        else:
            # Check if radius must be shrunk
            radius_list.append(radius)
            if fail_i % evals_shrink == 0:
                radius = radius*shrink_ratio
                r      = np.array([param_min, param_max])*radius

            # Sample new parameter values
            param = sample_local_params(best_param, r[1], r[0])

        # STEP 3: Construct the policy network with the sampled parameters
        policy_net.load_state_dict(param)

        # STEP 4: Evaluate the policy
        reward, std = reward_fcn(policy_net, env, num_episodes=num_episodes_avg)
        
        # STEP 5: Check if new policy is better than the best one found so far
        if reward > best_reward:
            # Update the new best policy
            best_reward = reward
            best_param = copy.deepcopy(param)
            # Save policy
            torch.save(policy_net.state_dict(), save_path)
        elif (reward < best_reward) and (i > iter_rs):
            fail_i += 1

        # STEP 6: Update iteration counter
        i += 1

        # Store the data for plotting
        reward_history.append(reward)
        std_history.append(std)
        best_reward_history.append(best_reward)

    # ######################################################################################
    # DO NOT MODIFY CODE BELOW THIS LINE
    # ######################################################################################

        # Check time
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    # Pack data for plotting
    plot_data = {'reward_history': reward_history,
                'std_history': std_history,
                'best_reward_history': best_reward_history}

    print(f"Policy model weights saved in: {save_path}")
    print(f"Best reward found during training: {best_reward}")
            
    return save_path , plot_data

def sample_uniform_params(params_prev, param_min, param_max):
    '''
    Sample random point within given parameter bounds. Tailored for EXPLORATORY purposes
    '''
    params = {k: torch.rand(v.shape) * (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_min, param_max):
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
