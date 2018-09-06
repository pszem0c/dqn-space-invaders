import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random

import warnings
warnings.filterwarnings('ignore')

env = retro.make(game="SpaceInvades-Atari2600")

print("The size of our frame is: {}".format(env.observation_space))
print("The action size is: {}".format(env.action_space.n))

possible_actions = np.array(np.identity(env.action_space.n, dtype=int).toList())
stack_size = 4
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def preprocess_frame(frame):
    gray = rgb2gray(frame)

    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    return stack_frames


    
