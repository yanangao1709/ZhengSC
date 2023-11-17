import gym
import numpy as np

env = gym.make("MountainCar-v0")  # 构建实验环境


class BespokeAgent:  # 自定义的智能体类
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):  # 学习
        pass


def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0.
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward


agent = BespokeAgent(env)
# env.reset(seed=0)  # 重置一个回合，设置了一个随机数种子，只是为了方便复现实验结果，通常可以把seed=0删掉
env.reset()

episode_reward = play_montecarlo(env, agent, render=True)
print("回合奖励={}".format(episode_reward))

episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]  # 为了评估智能体性能求出连续交互100回合的平均奖励
print("平均回合奖励={}".format(np.mean(episode_rewards)))
env.close()  # 关闭环境
