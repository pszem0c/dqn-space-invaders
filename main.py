import tensorflow as tf
import numpy as np
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

import random

import warnings

import memory
import dqnetwork

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('spaceinvaders.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

warnings.filterwarnings('ignore')

env = retro.make(game="SpaceInvaders-Atari2600")

logger.info("The size of our frame is: {}".format(env.observation_space))
logger.info("The action size is: {}".format(env.action_space.n))

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
memory_size = 30000

### preprocessing parameters
stack_size = 4

### training
training = True

### render
episode_render = False

### play agent
agent_test = False


possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

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
   
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

tf.reset_default_graph()
DQNetwork = dqnetwork.DQNetwork(state_size, action_size, learning_rate)
dqn_memory = memory.Memory(max_size = memory_size)

for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    choice = random.randint(1, len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)

    #env.render()

    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

    if done:
        next_state = np.zeros(state.shape)
        dqn_memory.add((state, action, reward, next_state, done))
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        dqn_memory.add((state, action, reward, next_state, done))
        state = next_state

writer = tf.summary.FileWriter("/tmp/tb/dqn/1")
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        choice = random.randint(1, len(possible_actions)) - 1
        action =possible_actions[choice]
    else:
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        decay_step = 0
        rewards_list = []
        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1
                decay_step += 1
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                        decay_step, state, possible_actions)
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps
                    total_reward = np.sum(episode_rewards)
                    rewards_list.append((episode, total_reward))
                    dqn_memory.add((state, action, reward, next_state, done))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    dqn_memory.add((state, action, reward, next_state, done))
                    state = next_state 
            ### LEARNING PART
            batch = dqn_memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []

            Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
            for i in range(0, len(batch)):
                terminal = dones_mb[i]
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamma*np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_mb = np.array([each for each in target_Qs_batch])
            loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                    feed_dict={DQNetwork.inputs_: states_mb,
                        DQNetwork.targetQ: targets_mb,
                        DQNetwork.actions_: actions_mb})
            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                        DQNetwork.targetQ: targets_mb,
                        DQNetwork.actions_: actions_mb})
            
            logger.info("Episode: {},Total reward: {},Explore P: {:.4f},Training Loss: {:.4f}".format(episode, total_reward, explore_probability, loss))
            writer.add_summary(summary, episode)
            writer.flush()

            if episode % 5 == 0:
                save_path = saver.save(sess, ".models/model.ckpt")
                logger.info("Model saved.")


if agent_test:
    total_test_rewards = []
    saver.restore(sess, ".models/model.ckpt")
    for episode in range(1):
        total_reward = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        logger.info("*******************************************************")
        logger.info("EPISODE ", episode)

        while True:
            state = state.reshape((1, *state_size))

            Qs = sess.run(DQNetwork.output, feed_dict= {DQNetwork.inputs_: state})

            choice = np.argmax(Qs)
            action = possible_actions[choice]

            next_state, reward, done, _ = env.step(action)
            env.render()

            total_reward += reward
            if done:
                logger.info("Score: {}". format(total_reward))
                total_test_rewards.append(total_reward)
                break

        state, stacked_frames = stack_frames(stacked_frames, state, False)
        state = next_state
    env.close()
