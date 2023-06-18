# reinforcement-learning
As someone who is interested in the studying robotics and AI, I am fascinated by the theory behind reinforcement learning. Reinforcement has many applications in robotics from behavioral learning to controls, and this is my repo to expirament with different methods and applications of reinforcement learning.

## Q-learning
The first method I implemented was Q-learning. Q-learning involves updating a table of (state,action) pairs with a function of the reward for taking that action at a given state. The Q-table is trained until the optimal weights are achieved to tell the agent what to do at any given action, which is the optimal policy.

Output of the [taxi](https://www.gymlibrary.dev/environments/toy_text/taxi/) environment from OpenAI gym, trained using Q-learning:
![image](https://github.com/OLeather/reinforcement-learning/assets/43189206/23b8e8f8-699d-4f1a-83c2-3a6db9dc58c2)

## Deep Q-learning
The second method I implemented was Deep Q-learning. I followed the [tutorial from PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) to learn about the method of Deep-Q learning. Deep Q-learning uses the same approach as Q-learning, but uses a neural network to train the Q table weights instead. It allows for training much more advanced scenarios, such as the [CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) environment. I chose to tackle the CartPole environment because control of a pendulum on a cart has been an ongoing project for me when learning about modern control theory, as seen in my [controls-project](https://github.com/OLeather?tab=repositories) repo.

The Deep-Q neural network is 3 layers, with an input layer that takes in the number of states, an intermediate 128x128 layer, and an output layer which outputs the Q-value for each action.
```python
def __init__(self, state_size, action_size):
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(state_size, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, action_size)

def forward(self, x):
    # print(x, self.layer1)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)
```

Training data for the Deep Q-learning neural network:
![Screenshot 2023-06-18 164437](https://github.com/OLeather/reinforcement-learning/assets/43189206/bd576abf-42c7-44dc-8ef8-859eb634ec71)
