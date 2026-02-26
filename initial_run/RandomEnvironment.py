import crafter
import numpy as np

def create_environment(log_dir='./logs'):
    """
    Initializes the Crafter environment and wraps it with a Recorder.
    """
    env = crafter.Env()
    env = crafter.Recorder(env, log_dir)
    return env

def run_episode(env, policy):
    """
    Runs one episode using the given policy.
    Logs step rewards, actions, and a combined dictionary of reward channels.
    Returns cumulative reward, list of step rewards, reward decomposition log, time steps, 
    the final observation, and the action log.
    """
    obs = env.reset()
    done = False
    cumulative_reward = 0
    reward_log = []            
    combined_reward_log = []   
    time_steps = []
    action_log = []            

    t = 0
    while not done:
        action = policy(env)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        reward_log.append(reward)
        time_steps.append(t)
        
        action_name = info.get('action_name', str(action))
        action_log.append(action_name)
        
        reward_channels = {}
        inventory = info.get('inventory', {})
        achievements = info.get('achievements', {})
        
        for key, value in inventory.items():
            reward_channels[key] = value
        for key, value in achievements.items():
            reward_channels[key] = value
        reward_channels['step_reward'] = info.get('reward', 0)
        
        combined_reward_log.append(reward_channels)
        t += 1
        
    return cumulative_reward, reward_log, combined_reward_log, time_steps, obs, action_log

def random_policy(env):
    """
    A simple random policy that selects an action uniformly at random.
    """
    return np.random.randint(env.action_space.n)
