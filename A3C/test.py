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
    parser.add_argument("--output_path", type=str, default="../sample")
    args = parser.parse_args()

    return args


def test(opt):
    torch.manual_seed(123)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
                                                    f"{opt.output_path}/video_{opt.world}_{opt.stage}.mp4")
    model = ActorCritic(num_states, num_actions)

    model.load_state_dict(torch.load(f"{opt.saved_path}/a3c_super_mario_bros_{opt.world}_{opt.stage}",
                                     map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True

    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()

        if info["flag_get"]:
            print(f"World {opt.world} stage {opt.stage} completed")
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
