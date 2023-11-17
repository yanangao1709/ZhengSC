import numpy as np
import pandas as pd

import quantum
from dqn import Dqn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


EPISODES = 1000
MEMORY_CAPACITY = 2000

env = quantum.Quantum()
NUM_STATES = env.num_states
NUM_ACTIONS = env.num_actions

def normalize(fidelity):
    b = np.array(fidelity)
    max_v = np.max(b)
    min_v = np.min(b)
    normalized_data = (b - min_v) / (max_v - min_v)
    return normalized_data

def main():
    net = Dqn(NUM_STATES, NUM_ACTIONS)
    print("The DQN is collecting experience...")
    acc_reward = []
    axx = []
    losses = []
    for episode in range(EPISODES):
        state = env.reset()
        step_counter = 0
        total_reward = 0
        while True:
            step_counter +=1
            # env.render()
            action = net.choose_action(state)
            next_state, reward, done = env.step(action, step_counter)
            net.store_trans(state, action, reward, next_state)#记录当前这组数据

            total_reward += reward
            if net.memory_counter >= MEMORY_CAPACITY: # 攒够数据一起学
                loss = net.learn()
                losses.append(loss)
            if done:
                # step_counter_list.append(step_counter)
                # net.plot(net.ax, step_counter_list)
                break
            state = next_state
        # print("episode {}, the reward is {}".format(episode, round(total_reward / step_counter, 3)))
        print(episode)
        # total_reward *= 14  # throughput scale
        # total_reward *= 10  # delay scale
        total_reward *= 16  # NoCRR throughput
        acc_reward.append(total_reward/step_counter)  # total_reward/step_counter
        axx.append(episode)
        net.plot(net.ax, acc_reward)

    plt.figure()
    x = [i for i in range(len(losses))]
    plt.plot(x, losses, "r-")
    plt.show()

    # fideity scale
    # acc_reward = normalize(acc_reward)

    plt.xlabel("episodes")
    plt.ylabel("throughput")
    plt.plot(axx, acc_reward, 'b-')
    plt.show()
    res = {"x":axx, "acc_reward": acc_reward}
    pd.DataFrame(res).to_csv('./OTiM2R/NoCRR-Throughput.csv', index=False)

if __name__ == '__main__':
    main()