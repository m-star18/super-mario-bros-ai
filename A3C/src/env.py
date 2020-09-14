import subprocess as sp

import cv2
import gym_super_mario_bros
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}X{height}",
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame

    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0

        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)

        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50

        return state, reward / 10., done, info
