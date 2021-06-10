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
from src.const import (
    ACTION_TYPE,
    LR,
    GAMMA,
    TAU,
    BETA,
    NUM_LOCAL_STEP,
    NUM_GLOBAL_STEP,
    NUM_PROCESS,
    SAVE_INTERVAL,
    MAX_ACTIONS,
    SEED,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default=ACTION_TYPE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=TAU, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=BETA, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=NUM_LOCAL_STEP)
    parser.add_argument("--num_global_steps", type=int, default=NUM_GLOBAL_STEP)
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESS)
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=MAX_ACTIONS, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="../trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    args = parser.parse_args()

    return args


def train(opt):
    torch.manual_seed(SEED)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    mp = _mp.get_context("spawn")
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(num_states, num_actions)
    global_model.share_memory()

    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1

        file_ = f"{opt.saved_path}/a3c_super_mario_bros_{previous_world}_{previous_stage}"
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []

    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))

        process.start()
        processes.append(process)

    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    process.start()
    processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
