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
