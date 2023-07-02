from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
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

# Set up a virtual display for rendering OpenAI gym environments.
print(tf.version.VERSION)

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

env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

train_env.reset()
eval_env.reset()

action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

q_network = sequential.Sequential([
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2.0, mode="fan_in", distribution='truncated_normal')),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2.0, mode="fan_in", distribution='truncated_normal')),
    tf.keras.layers.Dense(num_actions, activation=None, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03), bias_initializer=tf.keras.initializers.Constant(-0.2))]
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_network,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

def average_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):
    time_step = environment.reset()

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      total_return += time_step.reward

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

table = reverb.Table(
    'uniform_table',
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name='uniform_table',
    sequence_length=2,
    local_server=reverb_server
)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  'uniform_table',
  sequence_length=2
)

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
    [rb_observer],
    max_steps = initial_collect_steps
).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(3)

iterator = iter(dataset)
print(iterator)


agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

avg_return = average_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

time_step = train_py_env.reset()

collect_driver = py_driver.PyDriver(env, py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True), [rb_observer], max_steps=collect_steps_per_iteration)

saver = PolicySaver(agent.policy, batch_size=None)

for _ in range(num_iterations):
  time_step, _ = collect_driver.run(time_step)

  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = average_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
  
  if step % save_interval == 0:
    model_filename = "{0}_step{1}".format(env_name, step)
    print('step = {0}: Saving model: {1}'.format(step, model_filename))
    saver.save("./save/{0}/{1}".format(env_name, model_filename))

model_filename = "{0}_step{1}".format(env_name, agent.train_step_counter.numpy())
print("Training done. Saving model...")
saver.save("./save/{0}/{1}".format(env_name, model_filename))