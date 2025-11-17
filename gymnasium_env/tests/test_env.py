import gymnasium as gym
import gymnasium_env

# Test specific action sequences to verify behavior
env = gym.make("gymnasium_env/Vikavolt-v0")
obs, info = env.reset(seed=42)  # Use seed for reproducible testing

starting = obs['pos']
print(f"Starting position ", starting)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
    print(f"Action {action}: obs={obs}, reward={reward}")

env.close()

