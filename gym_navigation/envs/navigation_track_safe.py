from __future__ import annotations

import random
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

import omnisafe
from omnisafe.envs.core import CMDP, env_register
from gym_navigation.envs.navigation_track import NavigationTrack


@env_register
class NavigationTrackSafe(NavigationTrack, CMDP): # MRO matters here
    """
    A 'safe' variant of NavigationTrack that returns an additional cost signal
    if the agent is too close to a wall.

    Omnisafe expects:
    - A .step() method returning (obs, reward, cost, terminated, truncated, info).
    - A .reset() method returning (obs, info).
    - Tensors, not numpy arrays!
    - _support_envs listing custom environment ID.
    """

    # register environment ID
    _support_envs: ClassVar[list[str]] = ['NavigationTrackSafe-v0']

    need_auto_reset_wrapper: bool = True
    need_time_limit_wrapper: bool = True

    def __init__(self, env_id: str = "", **kwargs: dict[str, Any]) -> None:
        """OmniSafe will pass env_id and possibly other config in kwargs."""
        kwargs.pop('num_envs', None)
        kwargs.pop('device', None)

        self._count = 0

        # Omnisafe wants this attribute (CMDP base class references it)
        self._num_envs = 1

        # Omnisafe expects these properties:
        # - self._observation_space
        # - self._action_space

        NavigationTrack.__init__(self, **kwargs)

        # self._action_space = spaces.Box(
        #     low=-1.0, high=1.0, shape=self.action_space.shape, dtype=np.float32
        # )

        # self._observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=self.observation_space.shape, dtype=np.float32
        # )

    @property
    def max_episode_steps(self) -> int:
        """Required if you want the time-limit wrapper to auto-truncate after N steps."""
        return 200

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment one timestep.
        Must return (obs, reward, cost, terminated, truncated, info),
        each as a torch.Tensor except info which is a dict.
        """

        # convert incoming action from torch to numpy for the parent and clips it to the allowed action space
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, self.action_space.low, self.action_space.high)

        obs_np, reward_np, terminated, truncated, info = super().step(action_np)
        cost_value = self._calculate_distance_cost()

        # convert everything to torch tensors for omnisafe
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        reward = torch.as_tensor(reward_np, dtype=torch.float32)
        cost = torch.as_tensor(cost_value, dtype=torch.float32)
        terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool)

        return obs, reward, cost, terminated_tensor, truncated_tensor, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment for a new episode. Must return (obs, info).
        """
        if seed is not None:
            self.set_seed(seed)

        obs_np, info = super().reset(seed=seed)
        obs = torch.as_tensor(obs_np, dtype=torch.float32)

        self._count = 0
        return obs, info

    def _calculate_distance_cost(self) -> float:
        """
        Use sensor readings (self._ranges) from the parent NavigationTrack environment.
        If any sensor reads < 1.0, return cost=1.0; else 0.0.
        """
        if (self._ranges < 1.0).any():
            return 1.0
        return 0.0

    def set_seed(self, seed: int) -> None:
        """Set RNG seeds as needed."""
        random.seed(seed)
        np.random.seed(seed)

    def render(self) -> Any:
        """Optionally override if you want custom rendering."""
        return super().render()
    
    def close(self) -> None:
        super().close()
    
    @property
    def action_space(self):
        return self._action_space
