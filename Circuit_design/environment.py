import numpy as np

class Environment:
    def __init__(self, grid_size=5, explo_reward=10, explo_penalty=0, shift_reward=20, shift_penalty=-5,
                 shift_interval=50):
        self.grid_size = grid_size
        self.explo_reward = explo_reward
        self.explo_penalty = explo_penalty
        self.shift_reward = shift_reward
        self.shift_penalty = shift_penalty
        self.shift_interval = shift_interval
        self.steps = 0
        self.state = (0, 0)
        self.region_boundary = grid_size // 2

    def get_region(self, state):
        _, col = state
        if col < self.region_boundary:
            return 0
        else:
            return 1

    def reset(self):
        self.steps = 0
        self.state = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.grid_size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.grid_size - 1:
            col += 1

        self.state = (row, col)
        self.steps += 1
        region = self.get_region(self.state)

        if region == 0:
            reward = self.explo_reward
        else:
            if (self.steps // self.shift_interval) % 2 == 0:
                reward = self.shift_reward
            else:
                reward = self.shift_penalty

        done = (self.steps >= 100)
        return self.state, reward, done