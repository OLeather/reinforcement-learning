from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

import numpy as np
import PIL.Image
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import PolicySaver

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 600
# Setup plot to render env in matplotlib
plt.ion()
fig1, ax1 = plt.subplots()
array = np.zeros(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
axim1 = ax1.imshow(array, vmin=0, vmax=99)
del array

def render(img):
    axim1.set_data(img)
    fig1.canvas.flush_events()
    

num_iterations = 20000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000 

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 1
eval_interval = 1000
save_interval = 5000

env_name = 'CartPole-v1'

eval_py_env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

eval_env.reset()

policy_filename = "save/CartPole-v1/CartPole-v1_step20000"
saved_policy = tf.compat.v2.saved_model.load(policy_filename)
policy_state = saved_policy.get_initial_state(batch_size=3)
time_step = eval_env.reset()
time_count = 0
while not time_step.is_last():
  policy_step = saved_policy.action(time_step, policy_state)
  policy_state = policy_step.state
  time_step = eval_env.step(policy_step.action)
  time_count += 1

print("Evaluation finished with a final score of:", time_count, "time steps!")

