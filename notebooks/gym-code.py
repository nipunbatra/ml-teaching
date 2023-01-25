import gym
import time
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

for j in range(2):
    print("*"*20)
    print(j )
    done = False

    while not done:
        action = np.random.choice([0, 1, 2])
        new_state, reward, done, info = env.step(action)
        env.render()
        print(info)
        #print(new_state, reward, action)
        time.sleep(0.001)

env.close()