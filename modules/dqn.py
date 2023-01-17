import os
import random
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
import gymnasium as gym
from collections import deque
from tqdm import trange

import warnings
warnings.filterwarnings("ignore")


class DQNAgent:
    """Implements a Deep Q-Network with replay memory and target network.
    """

    def __init__(self, env: gym.Env, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float,
                 lr: float = 1e-4, C: int = 10, batch_size: int = 128, memory_size: int = 10000) -> None:
        """Initialize agent.

        Args:
            env (gym.Env): An OpenAI Gym environment.
            gamma (float): Discount factor for the returns.
            epsilon_init (float): Starting value for exploration rate.
            epsilon_min (float): Minimum value for exploration rate.
            epsilon_decay (float): Percentage of episodes to decay epsilon from initial to minimum.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            C (int, optional): Target network update frequency (in steps). Defaults to 10.
            batch_size (int, optional): Size of the batch sampled from the memory. Defaults to 128.
            memory_size (int, optional): Capacity of the replay memory. Defaults to 10000.
        """
        self.env = env
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.C = C
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay

        self.n_frames = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory = ReplayMemory(memory_size)
        self.model = RacingNet(self.n_frames, 5)
        self.model.to(self.device)
        self.target = RacingNet(self.n_frames, 5)
        self.target.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()

    def act(self, obs: torch.Tensor, epsilon: float = 0.0) -> int:
        """Choose optimal action given an observation with probability of 1 - epsilon.

        Args:
            obs (torch.Tensor): An observation from the environment.
            epsilon (float, optional): The probability of choosing a random action. Defaults to 0.0.

        Returns:
            int: Action to take.
        """
        if torch.rand(1) < epsilon:
            action = torch.randint(high=5, size=(1,))
        else:
            logits = self.model(obs.unsqueeze(0).to(self.device))
            action = logits.view(-1).argmax(0)

        return int(action.item())

    def play(self, n_episodes: int) -> dict:
        """Play the environment a given number of episodes.

        Args:
            n_episodes (int): Number of episodes.

        Returns:
            dict: Dictionary with played episodes and the corresponding scores.
        """
        print(f"### Device: {self.device} ###")

        results = {"episode": [], "score": []}
        frames = deque(maxlen=self.n_frames)

        pbar = trange(n_episodes)
        for episode in pbar:
            frame, _ = self.env.reset()
            for _ in range(self.n_frames):
                frames.append(self.transfrom_frame(frame))

            score = 0.0

            while True:
                obs = torch.vstack(list(frames))
                action = self.act(obs)
                frame, reward, terminated, truncated, _ = self.env.step(action)
                frames.append(self.transfrom_frame(frame))

                score += float(reward)

                if terminated or truncated:
                    break

            # Update stats and progress bar
            results["episode"].append(episode+1)
            results["score"].append(score)
            pbar.set_description(f"Score={score:.0f}")

        self.env.close()

        return results

    def train(self, n_episodes: int, save_every: int = 100) -> dict:
        """Train the agent for given number of episodes.

        Args:
            n_episodes (int): Number of episodes.
            save_every (int, optional): Interval for saving the model. Defaults to 100.

        Returns:
            dict: Dictionary with training episodes and the corresponding scores.
        """
        print(f"### Device: {self.device} ###")

        results = {"episode": [], "score": []}
        frames = deque(maxlen=self.n_frames)
        epsilons = self.schedule_decay(n_episodes, self.epsilon_init, self.epsilon_min, self.epsilon_decay)
        steps = 0

        pbar = trange(n_episodes)
        for episode in pbar:
            # Initialize the first sequence of frames, i.e. copy the first frame till the deque is full
            frame, _ = self.env.reset()
            for _ in range(self.n_frames):
                frames.append(self.transfrom_frame(frame))

            score = 0.0

            while True:
                # Choose action and observe next state
                obs = torch.vstack(list(frames))
                action = self.act(obs, epsilons[episode])
                frame, reward, terminated, truncated, _ = self.env.step(action)

                # Append frame and create new observation
                frames.append(self.transfrom_frame(frame))
                obs_ = torch.vstack(list(frames))

                # Add experience to replay memory
                self.memory.remember((obs, action, float(reward), obs_, terminated))

                # Optimize agent
                if episode > 0:
                    self.optimize()

                # Update target network every <C> steps
                if steps % self.C == 0:
                    self.target.load_state_dict(self.model.state_dict())

                obs = obs_
                steps += 1
                score += float(reward)

                if terminated or truncated:
                    break

            # Update stats and progress bar
            results["episode"].append(episode+1)
            results["score"].append(score)
            pbar.set_description(f"Score={score:.0f}")

        self.env.close()

        return results

    def optimize(self) -> None:
        """Sample a batch from memory and update network by gradient descent.
        """
        obs, action, reward, obs_, terminated = self.memory.sample(self.batch_size)

        # Put tensors to device
        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        obs_ = obs_.to(self.device)
        terminated = terminated.to(self.device)

        # Compute loss
        qs = self.model(obs).gather(1, action)
        max_qs_ = self.target(obs_).max(1, keepdim=True)[0]
        td_error = reward + self.gamma * max_qs_ * (1 - terminated)
        loss = self.criterion(td_error, qs)

        # Make gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transfrom_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert a single frame to grayscale and crop it to 84x84.

        Args:
            frame (np.ndarray): Single frame in the form of width x height x channel.

        Returns:
            torch.Tensor: The preprocessed frame.
        """
        image_tensor = F.to_tensor(frame)
        image_tensor = F.rgb_to_grayscale(image_tensor)
        image_tensor = F.resized_crop(image_tensor, 0, 0, 84, 96, [84, 84])

        return image_tensor

    def schedule_decay(self, n_episodes: int, init_value: float, min_value: float,
                       decay_ratio: float) -> np.ndarray:
        """Compute decaying values for all episodes in advance.
        Args:
            n_episodes (int): Number of episodes.
            init_value (float): Starting value.
            min_value (float): Smallest value. Must be within [init_value, 0.0).
            decay_ratio (float): Percentage of steps to decay the values from initial to minimum.
        Returns:
            np.ndarray: Values for all episodes.
        """
        steps = int(n_episodes * decay_ratio)
        epsilons = np.concatenate([
            np.geomspace(start=init_value, stop=min_value, num=steps),
            np.full(n_episodes - steps, min_value)
        ])

        return epsilons

    def save_model(self, dir: str, name: str) -> None:
        """Save model.
        Args:
            dir (str): Directory
            name (str): File name
        """
        file = os.path.join(os.getcwd(), dir, name + ".pt")
        torch.save(self.model.state_dict(), file)

    def load_model(self, dir: str, name: str) -> None:
        """Load model.
        Args:
            dir (str): Directory
            name (str): File name
        """
        file = os.path.join(os.getcwd(), dir, name + ".pt")
        self.model.load_state_dict(torch.load(file, map_location=self.device))


class RacingNet(nn.Module):
    """Implements a CNN.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the network.

        Args:
            input_dim (int): Number of input channels.
            output_dim (int): Number of output neurons.
        """
        super(RacingNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        return self.layers(x)


class ReplayMemory:
    """Memory for storing the experience tuples.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize memory.

        Args:
            capacity (int): Maximum capacity of the memory.
        """
        self.experiences = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.experiences)

    def remember(self, experience: tuple[torch.Tensor, int, float, torch.Tensor, bool]) -> None:
        """Save an experience tuple from the environment in the memory.

        Args:
            experience (tuple[torch.Tensor, int, float, torch.Tensor, bool]): (obs, action, reward, obs_, terminated).
        """
        self.experiences.append(experience)  # (obs, action, reward, obs_, terminated)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random batch from the memory.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            tuple[torch.Tensor, ...]: A random batch of experiences.
        """
        samples = random.choices(list(self.experiences), k=batch_size)
        obs, action, reward, obs_, terminated = zip(*samples)

        return (
            torch.stack(obs),
            torch.from_numpy(np.stack(action)).to(torch.int64).view(batch_size, -1),
            torch.tensor(reward).view(batch_size, -1),
            torch.stack(obs_),
            torch.tensor(terminated, dtype=torch.int64).view(batch_size, -1)
        )
