import numpy as np
import copy


def Path(states, actions, rewards, next_states, dones):
    return {"states" : np.array(states, dtype=np.float32),
            "rewards" : np.array(rewards, dtype=np.float32),
            "actions" : np.array(actions, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32)}

def get_pathlength(path):
    return len(path["rewards"])

def sample_trajectory(env, policy):
    # TODO: get this from hw1 or hw2
    # initialize env for the beginning of a new rollout
    state = env.reset() # HINT: should be the output of resetting the env

    # init vars
    states, actions, rewards, next_states, dones = [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        states.append(state)
        action = policy.get_action(state) # HINT: query the policy's get_action function
        action = action[0]
        actions.append(action)

        # take that action and record results
        state, reward, done = env.step(action)

        # record result of taking that action
        steps += 1
        next_states.append(state)
        rewards.append(reward)
        dones.append(done)

        if done:
            break

    return Path(states, actions, rewards, next_states, dones)

def sample_trajectories(env, policy, min_timesteps_per_batch):
    """
        Collect rollouts using policy
        until we have collected min_timesteps_per_batch steps
    """
    # TODO: get this from hw1 or hw2

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy)
        timesteps_this_batch += get_pathlength(path=path)
        paths.append(path)

    return paths

def sample_n_trajectories(env, policy, ntraj):
    # TODO: get this from hw1
    paths = []

    for _ in range(ntraj):
        path = sample_trajectory(env=env, policy=policy)
        paths.append(path)

    return paths

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    states = np.concatenate([path["states"] for path in paths])
    actions = np.concatenate([path["actions"] for path in paths])
    next_states = np.concatenate([path["next_states"] for path in paths])
    dones = np.concatenate([path["dones"] for path in paths])
    concatenated_rewards = np.concatenate([path["rewards"] for path in paths])
    unconcatenated_rewards = [path["rewards"] for path in paths]
    return states, actions, next_states, dones, concatenated_rewards, unconcatenated_rewards

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data