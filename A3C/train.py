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
