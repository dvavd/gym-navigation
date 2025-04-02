import numpy as np
from gym_navigation.envs.navigation_track import NavigationTrack

class NavigationTrackSafe(NavigationTrack):
    """
    A 'safe' variant of NavigationTrack that returns an additional cost signal
    if the agent is too close to a wall.
    """

    # Distance threshold for safety (in same units as _ranges).
    _SAFETY_THRESHOLD = 1.0

    def step(self, action):
        # perform the usual NavigationTrack step
        obs, reward, terminated, truncated, info = super().step(action)

        # compute safety cost (0 if safe, >0 if too close)
        cost = self._calculate_distance_cost()
        info["cost"] = cost

        return obs, reward, terminated, truncated, info

    def _calculate_distance_cost(self) -> float:
        """
        Uses sensor readings (self._ranges) from NavigationTrack.
        If any reading is below _SAFETY_THRESHOLD, we set cost=1.0.
        """
        # If any sensor sees a distance < SAFETY_THRESHOLD => cost=1.0
        if (self._ranges < self._SAFETY_THRESHOLD).any():
            return 1.0
        return 0.0