import gymnasium as gym

# Create the environment
env = gym.make('gym_navigation:NavigationTrack-v0', render_mode='human', track_id=1)

# Reset the environment
observation, info = env.reset()

try:
    for _ in range(1000):
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()

finally:
    env.close()
