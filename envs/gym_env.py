import gymnasium as gym
import torch
from envs.base_env import BaseEnv, DoneFlags

class GymEnv(BaseEnv):
    NAME = "gym"

    def __init__(self, config, device, visualize):
        # call BaseEnv constructor (sets mode/visualize flag)
        super().__init__(visualize)

        self._device = device
        self._time_limit = config.get("time_limit", None)

        # cfg["env_name"] will be like "gym:CartPole-v1"
        env_spec = config["env_name"][len("gym:"):]  # strip "gym:"
        self._env = gym.make(env_spec)
        self._action_space = self._env.action_space

        # initialize observation
        self.reset()

    def reset(self):
        obs, _ = self._env.reset()
        # convert numpy observation to torch tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self._device)
        return obs_tensor, {}

    def step(self, action):
        # convert torch action to numpy
        act_np = action.cpu().numpy()
        next_obs, reward, terminated, truncated, info = self._env.step(act_np)
        done_flag = DoneFlags.NULL.value

        if terminated:
            done_flag = DoneFlags.FAIL.value
        elif truncated or (self._time_limit and self._env.unwrapped.env_step >= self._time_limit):
            done_flag = DoneFlags.TIME.value

        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self._device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self._device)
        done_tensor = torch.tensor([done_flag], dtype=torch.int, device=self._device)

        return next_obs_tensor, reward_tensor, done_tensor, info

