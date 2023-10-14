from MLP_policy import MLPPolicy
from bootstrapped_continuous_critic import BootstrappedContinuousCritic
from replay_buffer import ReplayBuffer
from utils import *
from Pendulum_model import PendulumModel
import torch
import pytorch_util as ptu
from logger import Logger
from collections import OrderedDict
import time
import os
import matplotlib.pyplot as plt
class PPO:
    def __init__(self, logdir, save_actor_model_path, save_critic_model_path) -> None:
        ptu.init_gpu()
        self.env = PendulumModel()
        self.actor = MLPPolicy(self.env.action_dim, self.env.state_dim, 1, 128, 1e-3)
        self.critic = BootstrappedContinuousCritic(self.env.action_dim, self.env.state_dim, 1, 128, 1e-2, 0.98)
        self.batchsize = 1000
        self.replay_buffer = ReplayBuffer()
        self.epochs = 50
        self.eps = 0.2
        self.n_iter = 100
        self.logger = Logger(logdir)

        self.save_actor_model_path = save_actor_model_path
        self.save_critic_model_path = save_critic_model_path

    def load_model(self, load_actor_model_path, load_critic_model_path, num):
        self.actor.load_state_dict(torch.load(load_actor_model_path+'/%d.pt' %num))
        self.critic.load_state_dict(torch.load(load_critic_model_path+'/%d.pt' %num))

    def actor_update(self, states:np.ndarray, actions:np.ndarray, rewards:np.ndarray, dones:np.ndarray):
        action_distr = self.actor.forward(ptu.from_numpy(states))
        old_log_probs = action_distr.log_prob(ptu.from_numpy(actions)).detach()
        advantages = self.critic.estimate_advantage(states, rewards, dones)
        advantages = ptu.from_numpy(advantages)

        for _ in range(self.epochs):
            action_distr = self.actor.forward(ptu.from_numpy(states))
            log_probs = action_distr.log_prob(ptu.from_numpy(actions))
            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()        

    def perform_logging(self, itr, eval_paths):
        eval_returns = [eval_path["rewards"].sum() for eval_path in eval_paths]
        eval_ep_lens = [len(eval_path["rewards"]) for eval_path in eval_paths]

        logs = OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_MinEpLen"] = np.min(eval_ep_lens)
        logs["Eval_MaxEpLen"] = np.max(eval_ep_lens)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
        logs["TimeSinceStart"] = time.time() - self.start_time
        # perform the logging
        for key, value in logs.items():
            # print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, itr)

        self.logger.flush()

    def train_loop(self):
        self.start_time = time.time()
        for itr in range(self.n_iter):
            print("collecting train data...\n")
            path = sample_trajectories(self.env, self.actor, self.batchsize)
            self.replay_buffer.add_rollouts(path)
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_recent_data(self.batchsize)

            print("Training model...\n")
            self.critic.update(states, next_states, rewards, dones)
            self.actor_update(states, actions, rewards, dones)

            print("collecting eval data...\n")
            eval_paths = sample_n_trajectories(self.env, self.actor, 50)
            print("Perform the logging...\n")
            self.perform_logging(itr, eval_paths)

            if(itr%10 == 0):
                print("Saving model...\n")
                self.actor.save(self.save_actor_model_path+'/%d.pt' %itr)
                self.critic.save(self.save_critic_model_path+'/%d.pt' %itr)
    
    def eval_plot(self):
        paths = sample_n_trajectories(self.env, self.actor, 2)
        for i in range(len(paths)):
            states = paths[i]["states"]
            fig, axes = plt.subplots(states.shape[1],1)
            for j in range(states.shape[1]):
                axes[j].plot(states[:,j])
        plt.show()
            


load_model = True
load_actor_model_path = "/home/jiao/rl_ws/test/model/actor_model10-10-2023_11-06-04"
load_critic_model_path = "/home/jiao/rl_ws/test/model/critic_model10-10-2023_11-06-04"
if __name__ == "__main__":

    log_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './log_data')
    if not (os.path.exists(log_data_path)):
        os.makedirs(log_data_path)
    logdir = "Pendulum_model" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(log_data_path, logdir)
    
    save_actor_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './model/actor_model') + time.strftime("%d-%m-%Y_%H-%M-%S")
    save_critic_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './model/critic_model') + time.strftime("%d-%m-%Y_%H-%M-%S")
    if not (os.path.exists(save_actor_model_path)):
        os.makedirs(save_actor_model_path)
    if not (os.path.exists(save_critic_model_path)):
        os.makedirs(save_critic_model_path)
    ppo = PPO(logdir, save_actor_model_path, save_critic_model_path)

    if load_model:
        ppo.load_model(load_actor_model_path, load_critic_model_path, 20)

    ppo.eval_plot()
    # ppo.train_loop()

