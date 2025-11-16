import gymnasium as gym
import gymnasium_env

# Test specific action sequences to verify behavior
env = gym.make("gymnasium_env/Vikavolt-v0")
obs, info = env.reset(seed=42)  # Use seed for reproducible testing

print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
for action in actions:
    old_pos = obs['agent'].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs['agent']
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")
