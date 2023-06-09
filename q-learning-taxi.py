import gym
import numpy as np
import random
import os
clear = lambda: os.system('clear')

def update_qtable(qtable, state, action, new_state, reward, learning_rate, discount_rate):
    qtable[state, action] += learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state, action])
    return qtable


def explore_exploit(env, qtable, state, epsilon, decay_rate):
    if random.uniform(0, 1) < epsilon:
        # explore
        action = env.action_space.sample()
    else:
        action = pick_optimal_action(qtable, state)

    return action


def pick_optimal_action(qtable, state):
    return np.argmax(qtable[state, :])


learning_rate = 0.9
discount_rate = 0.8

epsilon = 1.0 # probability that our agent will explore
decay_rate = 0.01 # decay rate of epsilon

training_episodes = 1000
max_steps = 99

env = gym.make('Taxi-v3', render_mode="ansi")

state_size = env.observation_space.n
action_size = env.action_space.n

qtable = np.zeros((state_size, action_size))


for e in range(training_episodes):
    state, _ = env.reset()

    for s in range(max_steps+1):
        # Pick action using explore exploit
        action = explore_exploit(env, qtable, state, epsilon, decay_rate)
        
        # step the environment
        new_state, reward, terminated, truncated, info = env.step(action)
        
        # update the qtable using q learning algorithm
        qtable = update_qtable(qtable, state, action, new_state, reward, learning_rate, discount_rate)

        state = new_state

        if(terminated or truncated):
            break
    
    # epsilon decreases exponentially -> our agent will explore less and less
    epsilon = np.exp(-decay_rate*epsilon)
    
    clear()
    print(f"Training epsiode {e}/{training_episodes}")

test_environments = 5
for i in range(1, test_environments+1):
    print(f"###### Environment {i}/{test_environments} ######")       
    state, _ = env.reset()

    rewards = 0

    # watch trained agent
    for s in range(max_steps+1):
        action = pick_optimal_action(qtable, state)
        new_state, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        print(f"step {s}/{max_steps} reward: {rewards}")
        print(env.render())

        state = new_state

        if(terminated or truncated):
            break