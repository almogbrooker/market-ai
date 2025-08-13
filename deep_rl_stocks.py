import numpy as np
import gym

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.action_space = gym.spaces.Discrete(3)  # buy, hold, sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        reward = np.random.randn()
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.df.iloc[self.current_step].values

def train_agent(env, timesteps=10000):
    print(f"Training for {timesteps} timesteps (dummy)...")
    return "trained_agent"

def evaluate_agent(agent, env):
    print("Evaluating agent (dummy)...")
