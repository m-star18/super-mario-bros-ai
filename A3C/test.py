import os

os.environ['OMP_NUM_THREADS'] = '1'

import argparse

import torch
import torch.nn.functional as F

from src.env import create_train_env
from src.model import ActorCritic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="../trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()

    return args
