import gymnasium as gym
import numpy as np
import random
import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import itertools
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 600
action_space    = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
        (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
        (-1, 0,   0), (0, 0,   0), (1, 0,   0)
    ]

# Setup plot to render env in matplotlib
plt.ion()
fig1, ax1 = plt.subplots()
array = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
axim1 = ax1.imshow(array, vmin=0, vmax=99)
del array

def render(img):
    axim1.set_data(img)
    fig1.canvas.flush_events()


def normalize_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))


def dqn(state_shape, action_shape, learning_rate=0.001):
    model = keras.Sequential()
    model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(216, activation='relu'))
    model.add(Dense(len(action_shape), activation=None))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, epsilon=1e-7))
    return model

def train_cycle(replay_memory, model, target_model, learning_rate = 1, discount_factor = 0.95, batch_size = 128, done = False, callbacks=[]):
    '''
    replay_memory : deque of transitions (observation, action, reward, new_observation, done)
    model : main model to train
    target_model : target model
    '''

    # get batch from random sample of replay memory
    batch = random.sample(replay_memory, batch_size)
    
    # get current states, current qs list, new current states, and future qs list
    current_states = np.array([transition[0] for transition in batch])
    current_qs_list = model.predict(current_states, verbose=0)
    new_current_states = np.array([transition[3] for transition in batch])
    future_qs_list = target_model.predict(new_current_states, verbose=0)

    # for each transition in the batch
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(batch):
        # calculate future qs and current qs using bellman eqn
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1-learning_rate) * current_qs[action] + learning_rate * max_future_q
        
        # add observation and updated current qs to X and Y list
        X.append(observation)
        Y.append(current_qs)
    
    # fit model to X and Y
    model.fit(np.array(X), np.array(Y), batch_size = batch_size, verbose = 0, shuffle = True, callbacks=callbacks) 

def epsilon_greedy(observation, epsilon, model):
    '''
    Returns the action to take
    '''
    if np.random.rand() <= epsilon:
        # print("picked random")
        # returns random action
        return random.randrange(len(action_space))
    else:
        prediction = model.predict(np.expand_dims(observation, axis=0), verbose=0)
        action = np.argmax(prediction[0])
        # print("picked model", prediction, action)
        # returns the action corresponding to the max q-value of the model prediction
        return action

def train(model, target_model, env, skip_frames = 2, episodes = 1000, memory_size = 5000, min_replay_size = 128, epsilon = 0.9, min_epsilon = 0.01, epsilon_decay = 0.005):
    replay_memory = deque(maxlen=memory_size)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train/new'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)


    update_target = 0
    total_steps = 0
    for episode in range(episodes):
        reward_sum = 0
        observation, _ = env.reset()
        observation = normalize_image(observation)
        state_frame_stack_queue = deque([observation]*3, maxlen=3)
        
        done = False
        total_time = 0
        negative_reward_counter = 0
        for sim_step in itertools.count():
            # if(episode % 10 == 0):
                # render env
            render(env.render())
            
            reward = 0    
            # choose action by epsilon greedy algorithm
            state = generate_state_frame_stack_from_queue(state_frame_stack_queue)
            action = epsilon_greedy(state, epsilon, model)

            # step environment given action to get new observation
            for _ in range(skip_frames+1):
                new_observation, r, truncated, terminated, info = env.step(action_space[action])
                done = terminated or truncated
                reward += r
                if done:
                    break
        
            # Extra bonus for the model if it uses full gas
            if action_space[action][1] == 1 and action_space[action][2] == 0:
                reward *= 1.5

            new_observation = normalize_image(new_observation)
            state_frame_stack_queue.append(new_observation)
            next_state = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            # add transition to replay memory
            replay_memory.append([state, action, reward, next_state, done])

            # every 4 steps run a training iteration on the neural network
            if (sim_step % 4 == 0 or done) and len(replay_memory) > min_replay_size:
                train_cycle(replay_memory, model, target_model, done, callbacks=[tensorboard_callback])
            
            # observation = new_observation
            negative_reward_counter = negative_reward_counter + 1 if sim_step > 100 and reward < 0 else 0

            reward_sum += reward
            # print(reward_sum)
            update_target += 1
            total_steps += 1
            total_time = sim_step
            if done or reward_sum < 0 or negative_reward_counter >= 25 or sim_step > 10000:
                print('total_time = {}, step = {}/{}, final reward = {}, epsilon = {}'.format(total_time, episode, episodes, reward, epsilon))
                break
        if episode % 5 == 0:
            print('Copying main network weights to target network')
            target_model.set_weights(model.get_weights())
            update_target = 0
        if episode % 100 == 0:
            print('Saving weights at episode:', episode)
            target_model.save_weights("./save/car_racing/episode"+str(episode)+".h5")
        with train_summary_writer.as_default():
            tf.summary.scalar('total_time', total_time, step=episode)
        # decay epsilon
        epsilon = min_epsilon + (.9 - min_epsilon) * np.exp(-epsilon_decay * episode)



env = gym.make("CarRacing-v2", domain_randomize=True, render_mode="rgb_array")

# normal reset, this changes the colour scheme by default
env.reset()

print(env.observation_space.shape)

model = dqn(env.observation_space.shape, action_space)
target_model = dqn(env.observation_space.shape, action_space)
target_model.set_weights(model.get_weights())

train(model, target_model, env)


env.close()
