from env import Easy21Env
import numpy as np
if __name__ == '__main__':
    env = Easy21Env()
    done = False
    cnt = 0
    state = env.reset()
    print(state)
    while not done:
        action = np.random.choice([0, 1], p=[0.85,0.15])
        state, reward, done = env.step(action)
        print(action, state, reward, done)
        cnt += 1
    print("Count: ", cnt)
