# %%
import torch
import torch.nn as nn
from collections import deque, namedtuple
import torch.optim as optim
import random
import numpy as np
import copy
import torch.nn.functional as F
import os

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # sample such that it includes the done == true
        done_samples = [t for t in self.memory if t.done]
        non_done_samples = [t for t in self.memory if not t.done]
        done_count = min(len(done_samples), batch_size // 2)
        non_done_count = batch_size - done_count
        done_batch = random.sample(done_samples, done_count)
        non_done_batch = random.sample(non_done_samples, non_done_count)
        batch = done_batch + non_done_batch
        random.shuffle(batch)
        return batch

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = deque([], maxlen=self.capacity)


class DQNModel(nn.Module):
    def __init__(self, state_size, action_size) -> None:
        super(DQNModel, self).__init__()
        self.lin1 = nn.Linear(state_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu1(self.lin1(x))
        x = self.relu2(self.lin2(x))
        x = self.dropout1(x)
        x = self.lin3(x)
        x = self.relu3(x)
        return x


# %%


class DQNAgent:
    def __init__(self, state_size, action_size) -> None:
        super(DQNAgent, self).__init__()
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 500
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)
        self.model_path = "./model_state/"
        self.lr = 1e-4
        self.policy_net = DQNModel(state_size, action_size)
        self.target_net = DQNModel(state_size, action_size)
        self.replay_buffer = ReplayMemory(capacity=100000)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.tau = 1.0
        os.makedirs(self.model_path, exist_ok=True)

    def save_policy(self, extension=None):
        checkpoint = {
            "model_state_dict": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if extension is not None:
            path = self.model_path + str(extension)
        else:
            path = self.model_path + "agent_state_dict"

        torch.save(checkpoint, path)

    def load_policy(self, extension=None):
        if extension is not None:
            path = self.model_path + str(extension)
        else:
            path = self.model_path + "agent_state_dict"

        self.policy_net.load_state_dict(torch.load(path)["model_state_dict"])
        self.optimizer.load_state_dict(torch.load(path)["optimizer_state_dict"])

    def select_action(self, state):
        "following epsilon greedy approach"
        if np.random.rand() <= max(self.epsilon, self.epsilon_min):
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            self.q_value = self.policy_net(state).detach().numpy()
            return np.argmax(self.q_value)

    def update_target_net(self):
        # target_net_state_dict = self.target_net.state_dict()
        # policy_net_state_dict = self.policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[
        #         key
        #     ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        # self.target_net.load_state_dict(target_net_state_dict)

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, epochs=1, target_update=False):
        if len(self.replay_buffer) < self.batch_size:
            return

        for epoch in range(epochs):
            transitions = self.replay_buffer.sample(batch_size=self.batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32)
            action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64)
            reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32)
            next_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
            done_batch = torch.tensor(batch.done, dtype=torch.bool)

            state_actions_values = self.policy_net(state_batch).gather(
                1, action_batch.unsqueeze(1)
            )
            state_actions_values.squeeze_()
            with torch.no_grad():
                next_state_action_values = (
                    self.target_net(next_batch).max(1)[0].detach()
                )

            next_state_action_values[done_batch] = 0.0

            expected_state_action_values = (
                next_state_action_values * self.discount_factor
            ) + reward_batch

            loss = F.mse_loss(state_actions_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if target_update:
            self.update_target_net()


# %%

# %%
