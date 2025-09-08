import numpy as np
import copy
import time
from tqdm import tqdm
       
def q_learning_alg( env,*, 
                    max_episodes = 500 , # Total training episodes
                    max_time = 5 , # Maximum execution time
                    epsilon = 0.5 , # Probability of taking a random action
                    alpha = 0.0001, # Learning rate
                    gamma = 0.99, # Discount factor
                    seed = 0
                    ): 
    '''
    Implementation of tabular Q-learning algorithm.
    '''
    # Fix seed for reproducibility across runs
    np.random.seed(seed)

    # Create buffer to store plot data
    plot_data = {}
    plot_data["episode"] = []
    plot_data["total_reward"] = [] 

    # Start timer
    start_time = time.time()
    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------
    
    # Initialize Q-table
    action_size = env.action_space.n
    state_size  = env.observation_space.n
    Q_table = np.zeros((state_size,action_size))

    # Optimization loop
    for episode in tqdm(range(max_episodes)):
        # Reset environment
        env.reset()
        state = env.obs
        state_idx = env.observation_lookup[tuple(state)]
        done = False
        
        total_reward = 0

        while not done:
            # Chose action from behaviour policy
            action, action_idx = epsilon_greedy_policy(env, state_idx, Q_table, epsilon)
            # Take action
            next_state, reward, done , _ = env.step(action)
            next_state_idx = env.observation_lookup[tuple(next_state)]

            # Update Q_table
            _ , next_action_idx = greedy_policy(env, next_state_idx, Q_table)
            td_error = reward + gamma * Q_table[next_state_idx,next_action_idx] - Q_table[state_idx,action_idx]
            Q_table[state_idx,action_idx] += alpha * td_error

            # Update state
            state , state_idx = next_state , next_state_idx

    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
    # -----------------------------------------------------------------------------------
            # Update counter
            total_reward += reward

            # Check execution time
            if (time.time() - start_time) > max_time:
                print("Timeout reached.\n   Returning best Q(s,a) table found so far.")
                return Q_table , plot_data 
        
        # print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        plot_data["episode"].append(episode)
        plot_data["total_reward"].append(total_reward)

    return Q_table , plot_data

def evaluate_action_value_function(env, demand_dataset, Q_table, verbose = True):

    # Initialize buffer list to store results of each run
    reward_list = []

    # Run each episode and compute total undiscounted reward
    for scenario in demand_dataset:
        # Fix customer demand (if provided)
        env.demand_dataset = scenario

        # Reset environment before running each episode
        env.reset()
        done = False

        # Initialize reward counter
        total_reward = 0
        
        while not done:
            # Get current state and its index in the Q(s,a) table
            state = env.obs
            state_idx = env.observation_lookup[tuple(state)]

            # Get action according to the greedy policy
            action , _ = greedy_policy(env, state_idx, Q_table)
            
            # Interact with the environment to get reward and next state
            state , reward, done, _ = env.step(action)
            total_reward += reward
            
        reward_list.append(total_reward)

    # Compute mean and standard deviation
    mean_return = np.mean(reward_list)
    std_return = np.std(reward_list)

    if verbose:
        print("Performance of the inventory management policy in the test set:\n - Average reward: {:.0f}\n - Reward standard deviation: {:.2f}".format(np.mean(reward_list), np.std(reward_list)))
    
    return reward_list

def greedy_policy(env, state_idx, Q): 
    '''
    Select action according to greedy policy:
    argmax_a {Q(s,a)}
    '''
    # Get index of the optimal action
    action_idx = np.argmax(Q[state_idx,:])
    # Map action index to actual action value
    return np.array(list(env.action_lookup.keys())[action_idx]) , action_idx

def epsilon_greedy_policy(env, state, Q, epsilon):
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        action = env.sample_action()
        return action , env.action_lookup[tuple(action)]
    else:
        # Exploit: choose best action 
        return greedy_policy(env,state,Q)