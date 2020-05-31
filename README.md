[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

#### Setup and activate a python 3.6 environment

This notebook requires a python 3.6 environment which you can create using conda.  Please see the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) if you don't have conda installed on your system.

```bash
conda create --name drlnd python=3.6
source activate drlnd
```

#### Download the ML Agents environment

The `Navigation.ipynb` notebook in this project has been developed to run on linux.
Instructions are provided therein to download the NoVis version of the Banana ML Agents environment that is provided for this course.
This environment may also be downloaded and installed manually by following the instructions below:

1. Download the environment from the following [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip).

2. Place the file in the top level folder of this repository and unzip (or decompress) the file.

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!

### Report

The `Report.ipynb` notebook provides describes the learning algorithm used, the chosen hyperparametes and the model architecture for the neural network which was used along with ideas for future work.
