# check_taxi_env_version.py
import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3")
env.reset(seed=0)
print("Gymnasium version:", gym.__version__)

desc = np.asarray(env.unwrapped.desc)
print("desc type:", type(desc))
print("desc shape:", desc.shape)
print("desc dtype:", desc.dtype)

print("\nRaw desc:")
for row in desc:
    print("".join(ch.decode() if isinstance(ch, bytes) else chr(ch) for ch in row))
