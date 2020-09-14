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
