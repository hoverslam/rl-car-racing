# Car Racing

The **Car Racing** environment is one of the easiest control tasks to learn from pixels. The observation space consists of a top-down 96x96 RGB image. The reward is -0.1 every frame and +1 for every track tile visited.

The discrete action space has 5 values:
- 0: do nothing
- 1: steer left
- 2: steer right
- 3: gas
- 4: brake

For more information see https://gymnasium.farama.org/environments/box2d/car_racing/.


## Solution

The environment is solved with a Deep Q-Network. It consist of two convolutional layers and one fully-connected hidden layer (each with ReLU activation). The output layer has a size corresponding to the number of actions the agent can take.

<p align="center"><img src="img/racing_net.png?raw=true" height="300"></p>

The inputs to this neural net are images from the last 4 frames stacked together. Beforehand three pre-preprocessing steps are done:
1. Convert image to grayscale.
2. Crop the bottom part containing infos about points, action taken, etc.
3. Resize to a squared image of final size 84x84 pixels.

Reference: [V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller (2013) Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)


## Results

| Training                                                    | After 2000 episodes                                |
|:-----------------------------------------------------------:|:--------------------------------------------------:|
| <img src="img/dqn_training.png?raw=true" height="300">      | <img src="img/dqn.gif?raw=true" height="300">      |


The average score over 100 episodes on the trained agent is 702.81.


## How to

Install dependencies with `pip install -r requirements.txt`.

`main.py train <n>` Train for n episodes.

`main.py play <n> <render>` Let the agent play n episodes. Render the environment with True, otherwise False.


## Dependencies

- Python v3.10
- Gymnasium v0.27.0
- Numpy v1.24.1
- PyTorch v 1.13.1
- Matplotlib v3.6.3
- Tqdm v4.64.1
- Typer v0.7.0
