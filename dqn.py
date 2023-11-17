import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import random

#hyper parameters
EPSILON = 0.9
GAMMA = 0.9
LR = 0.005 # throughput
# LR = 0.05    # delay
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 8


class Net(nn.Module):
    def __init__(self,NUM_STATES, NUM_ACTIONS):
        # 在Net类中调用其父类的__init__方法。
        # 这是为了继承父类的属性和方法，并初始化父类中定义的变量或对象。
        # super()函数可以避免直接引用父类名，从而更加灵活和通用。
        super(Net, self).__init__()
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        self.set_seed(1)

        # self.fc1的输入维度为NUM_STATES，输出维度为30，
        self.input_layer = nn.Linear(self.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32,64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64,32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        # self.fc2的输入维度为30，输出维度为NUM_ACTIONS
        # self.fc_1 = nn.Linear(30, self.NUM_ACTIONS)
        # self.fc_1.weight.data.normal_(0, 0.1)
        #
        # self.fc_2 = nn.Linear(30, self.NUM_ACTIONS)
        # self.fc_2.weight.data.normal_(0, 0.1)
        #
        # self.fc_3 = nn.Linear(30, self.NUM_ACTIONS)
        # self.fc_3.weight.data.normal_(0, 0.1)
        #
        # self.fc_4 = nn.Linear(30, self.NUM_ACTIONS)
        # self.fc_4.weight.data.normal_(0, 0.1)

        self.fc_1 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_1.weight.data.normal_(0, 0.1)

        self.fc_2 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_2.weight.data.normal_(0, 0.1)

        self.fc_3 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_3.weight.data.normal_(0, 0.1)

        self.fc_4 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_4.weight.data.normal_(0, 0.1)

        self.fc_5 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_5.weight.data.normal_(0, 0.1)
        self.fc_6 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_6.weight.data.normal_(0, 0.1)
        self.fc_7 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_7.weight.data.normal_(0, 0.1)
        self.fc_8 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_8.weight.data.normal_(0, 0.1)
        self.fc_9 = nn.Linear(32, self.NUM_ACTIONS)
        self.fc_9.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))

        # x1 = self.fc_1(x)
        # x2 = self.fc_2(x)
        # x3 = self.fc_3(x)
        # x4 = self.fc_4(x)

        x1 = F.relu(self.fc_1(x))
        x2 = F.relu(self.fc_2(x))
        x3 = F.relu(self.fc_3(x))
        x4 = F.relu(self.fc_4(x))
        x5 = F.relu(self.fc_5(x))
        x6 = F.relu(self.fc_6(x))
        x7 = F.relu(self.fc_7(x))
        x8 = F.relu(self.fc_8(x))
        x9 = F.relu(self.fc_9(x))

        # return x1, x2, x3, x4
        return x1, x2, x3, x4, x5, x6, x7, x8, x9

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

class Dqn():
    def __init__(self, NUM_STATES, NUM_ACTIONS):
        self.NUM_STATES = NUM_STATES
        self.NUM_ACTIONS = NUM_ACTIONS
        # 主网络eval_net（Q function/Q-table）
        # target网络target_net，记忆
        # eval_net 用于评估当前状态和动作之间的 Q 值，
        # 而 target_net 用于评估下一个状态和动作之间的 Q 值
        self.eval_net, self.target_net = Net(self.NUM_STATES, NUM_ACTIONS), Net(self.NUM_STATES, NUM_ACTIONS)
        # 存数据
        # self.memory = np.zeros((MEMORY_CAPACITY, 17))
        self.memory = np.zeros((MEMORY_CAPACITY, 46))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR) # 优化器：针对主网络进行更新
        self.loss = nn.MSELoss() # 回归

        self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, action, [reward], next_state))#记录一条数据
        self.memory[index,] = trans
        self.memory_counter += 1

    def choose_action(self, state_para):
        action = []
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = torch.unsqueeze(torch.FloatTensor(state_para) ,0)
        if np.random.randn() <= EPSILON:
            action_value = self.eval_net.forward(state)
            for av in action_value:
                a = torch.max(av, 1)[1].data.item()
                action.append(a)
        else: # 随机
            for i in range(9):
                a = np.random.randint(0,self.NUM_ACTIONS)
                action.append(a)
        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("accumulated reward")
        ax.plot(x, 'b-')
        plt.pause(0.00001)
        if ax == 500:
            plt.show()

    def learn(self):
        # 每学习100次之后，重新对target网络赋值
        if self.learn_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())     #  学了100次之后target才更新（直接加载eval的权重）
        self.learn_counter+=1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)    # 获取一个batch数据

        batch_memory = self.memory[sample_index, :]

        batch_state = torch.FloatTensor(batch_memory[:, :self.NUM_STATES])
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, self.NUM_STATES:self.NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.NUM_STATES+1: self.NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.NUM_STATES:])

        q_eval_total = []
        for bs in self.eval_net(batch_state):
            q_eval_total.append(bs.gather(1, batch_action))
        q_eval = sum(q_eval_total)/len(q_eval_total)
        # q_eval = self.eval_net(batch_state).gather(1, batch_action) # 得到当前Q(s,a)

        q_next_total = []
        for bs in self.eval_net(batch_next_state):
            q_next_total.append(bs.gather(1, batch_action))
        q_next = sum(q_next_total) / len(q_next_total)
        # q_next = self.target_net(batch_next_state).detach() # 得到Q(s',a')，有三个值，下面选max
        q_target = batch_reward + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1) # bellman公式：Q=R+折扣*Q‘

        loss = self.loss(q_eval, q_target) # 差异越小越好
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # 梯度更新
        return loss