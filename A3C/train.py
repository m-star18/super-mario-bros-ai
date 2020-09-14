import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import shutil

import torch
import torch.multiprocessing as _mp

from src.env import create_train_env
from src.model import ActorCritic
from src.optimizer import GlobalAdam
from src.process import local_test, local_train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="../trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    args = parser.parse_args()

    return args
