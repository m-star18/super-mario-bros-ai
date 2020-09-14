import timeit
from collections import deque

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from src.env import create_train_env
from src.model import ActorCritic


def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()

    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.train()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    curr_episode = 0

    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           f"{opt.saved_path}/a3c_super_mario_bros_{opt.world}_{opt.stage}")
            print(f"Now Process {index}. Episode {curr_episode}")
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        log_policies = []
        values = []
        rewards = []
        entropies = []
