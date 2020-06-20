# Udacity Navigation Project Report
Luis Arias

*June 2020*

---

## Learning Algorithm

The algorithms used in this project are a combination of:

- Double DQN (see https://arxiv.org/abs/1509.06461)
- Dueling DQN (see https://arxiv.org/abs/1511.06581)

Double DQN was implemented in the agent's `learn` method (see [`agent.py`](./agent.py)) by predicting actions to be taken using the local network and predicting Q values for the next step using the target network as follows:

```python
        # Use Double DQN: Get predicted actions from local network model
        local_actions = self.qnetwork_local(
            next_states).detach().argmax(dim=1).unsqueeze(1)
        # Get predicted Q values (for next states) from target model using predicted actions
        Q_targets_next = self.qnetwork_target(
            next_states).gather(1, local_actions).detach()

```

Dueling DQN was implemented in the neural network model (see [`model.py`](./model.py)) by splitting the neural network into an advantage head and a value head in the `_init_` method as follows:

```python
        # trunk
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # adv head
        self.adv1 = nn.Linear(fc2_units, head_units)
        self.adv2 = nn.Linear(head_units, action_size)

        # val head
        self.val1 = nn.Linear(fc2_units, head_units)
        self.val2 = nn.Linear(head_units, 1)
```

In the `forward` method, the logits of the two heads are combined according to the Dueling DQN algorithm as follows:

```python
        return val.expand(adv.size()) + adv - adv.mean(dim=1, keepdim=True)
```

### Hyperparameters

Hyperparameters for the neural network model where chosen using the DQN exercise as a guide.  Here the number of units in the linear layers (64) was the square of the environment complexity (8). Thus given that the Banana environment size was reported to be 37, the number of units in the linear layers could have been 1369, however before attempting to use this model size, attempts were made at 768 and 1024 with significantly similar results.

## Methods

The approach to implementing these algorithms was to first implement them on a simpler environment, namely the [Deep Q-Network exercise](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/exercise) we studied in the Udacity Deep Reinforcement Learning course which uses the [OpenAI Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) environment.  This approach was important because the higher complexity ML Agents Bananas environment took a prohibitely long amount of compute time to solve on a CPU (6 to 13 hours).  Furthermore, using a High Performance GPU (Tesla V100 · 16 GB Memory - 61 GB RAM · 100 GB SSD) did not significantly improve performance.

The [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) algorithm was implemented and evaluated on OpenAI's Lunar Lander environment but was not retained for this project because in spite of providing slightly better efficiency in terms of episodes, the additional compute cost could have significantly increased the time to solve the Bananas environment.


## Results

## Future Work
