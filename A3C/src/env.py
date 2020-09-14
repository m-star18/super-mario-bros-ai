import subprocess as sp

import cv2
import gym_super_mario_bros
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import (COMPLEX_MOVEMENT, RIGHT_ONLY,
                                          SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace
