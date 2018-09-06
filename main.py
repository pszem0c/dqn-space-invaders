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
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

### model hyperparameters
state_size = [110, 84, 4]
action_size = env.action_space.n
learning_rate = 0.00025

### training parameters
total_episodes = 50
max_steps = 50000
batch_size = 64

### exploration parameters
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

### Q learning parameters
gamma = 0.9

### memory hyperparameters
pretrain_length = batch_size
memory_size = 1000000

### preprocessing parameters
stack_size = 4

### visualization
training = False

### render
episode_render = False

def preprocess_frame(frame):
    gray = rgb2gray(frame)

    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
   
        stacked_state = np.stack(stack_frames, axis=2)

    else:
        stack_frames.append(frame)
        stacked_state = np.stack(stack_frames, axis=2)

    return stacked_state, stack_frames
    
