import numpy as np

class Quantum():
    # 参数定义
    def __init__(self):
        # 在Net类中调用其父类的__init__方法。
        # 这是为了继承父类的属性和方法，并初始化父类中定义的变量或对象。
        # super()函数可以避免直接引用父类名，从而更加灵活和通用。
        super(Quantum, self).__init__()
        # self.destinations = self.read_destinations('')

        # 初始化当前状态
        self.current_node = None
        self.current_dest = None
        self.current_path = []

        self.num_actions = 3
        self.num_states = 18

        self.state = []
        self.reward = 0

        self.gamma = 0.99
        self.num_steps = 1000

    # reset 环境
    def reset(self):
        self.state = []
        self.reward = 0

        # 将网络拓扑信息编码为状态向量
        # self.r1_capacity = 6
        # self.r2_capacity = 6
        # self.s1_capacity = 6
        # self.d1_capacity = 6
        # self.s2_capacity = 6
        # self.d2_capacity = 6
        # self.state.append(self.r1_capacity)
        # self.state.append(self.r2_capacity)
        # self.state.append(self.s1_capacity)
        # self.state.append(self.d1_capacity)
        # self.state.append(self.s2_capacity)
        # self.state.append(self.d2_capacity)

        self.v1_cap = 2
        self.v2_cap = 3
        self.v3_cap = 3
        self.v4_cap = 4
        self.v5_cap = 6
        self.v6_cap = 4
        self.v7_cap = 2
        self.v8_cap = 5
        self.v9_cap = 1
        self.v10_cap = 3
        self.v11_cap = 7
        self.v12_cap = 3
        self.v13_cap = 2
        self.v14_cap = 4
        self.v15_cap = 2
        self.v16_cap = 4
        self.v17_cap = 1
        self.v18_cap = 2
        self.state.append(self.v1_cap)
        self.state.append(self.v2_cap)
        self.state.append(self.v3_cap)
        self.state.append(self.v4_cap)
        self.state.append(self.v5_cap)
        self.state.append(self.v6_cap)
        self.state.append(self.v7_cap)
        self.state.append(self.v8_cap)
        self.state.append(self.v9_cap)
        self.state.append(self.v10_cap)
        self.state.append(self.v11_cap)
        self.state.append(self.v12_cap)
        self.state.append(self.v13_cap)
        self.state.append(self.v14_cap)
        self.state.append(self.v15_cap)
        self.state.append(self.v16_cap)
        self.state.append(self.v17_cap)
        self.state.append(self.v18_cap)

        return self.state

    def step(self, action, step_counter):

        # 获取当前状态
        state = self.get_state()
        # 更新状态
        next_state = self.transmit(action)
        # 计算奖励
        reward = self.compute_reward(action)
        # 判断是否结束
        done = self.check_termination(step_counter)

        return next_state, reward, done

    def get_state(self):
        state = []
        # state.append(self.r1_capacity)
        # state.append(self.r2_capacity)
        # state.append(self.s1_capacity)
        # state.append(self.d1_capacity)
        # state.append(self.s2_capacity)
        # state.append(self.d2_capacity)

        state.append(self.v1_cap)
        state.append(self.v2_cap)
        state.append(self.v3_cap)
        state.append(self.v4_cap)
        state.append(self.v5_cap)
        state.append(self.v6_cap)
        state.append(self.v7_cap)
        state.append(self.v8_cap)
        state.append(self.v9_cap)
        state.append(self.v10_cap)
        state.append(self.v11_cap)
        state.append(self.v12_cap)
        state.append(self.v13_cap)
        state.append(self.v14_cap)
        state.append(self.v15_cap)
        state.append(self.v16_cap)
        state.append(self.v17_cap)
        state.append(self.v18_cap)
        return state

    def transmit(self, action):
        # 路径信息
        # if self.s1_capacity >0:
        #     self.s1_capacity -= action[0]
        #     self.s1_capacity -= action[1]
        #
        # if self.d1_capacity > 0:
        #     self.d1_capacity -= action[0]
        #     self.d1_capacity -= action[1]
        #
        # if self.r1_capacity > 0:
        #     self.r1_capacity -= action[1]
        #     self.r1_capacity -= action[2]
        #     self.r1_capacity -= action[3]
        #
        # if self.r2_capacity > 0:
        #     self.r2_capacity -= action[1]
        #     self.r2_capacity -= action[3]
        #
        # if self.s2_capacity > 0:
        #     self.s2_capacity -= action[2]
        #     self.s2_capacity -= action[3]
        #
        # if self.d2_capacity > 0:
        #     self.d2_capacity -= action[2]
        #     self.d2_capacity -= action[3]


        if self.v1_cap > 0:
            self.v1_cap -= action[0]
            self.v1_cap -= action[1]
            self.v1_cap -= action[2]
            self.v1_cap -= action[4]
        if self.v3_cap > 0:
            self.v3_cap -= action[2]
        if self.v4_cap > 0:
            self.v4_cap -= action[0]
            self.v4_cap -= action[1]
        if self.v5_cap > 0:
            self.v5_cap -= action[1]
            self.v5_cap -= action[2]
        if self.v6_cap > 0:
            self.v6_cap -= action[3]
            self.v6_cap -= action[4]
            self.v6_cap -= action[5]
        if self.v7_cap > 0:
            self.v7_cap -= action[3]
            self.v7_cap -= action[4]
            self.v7_cap -= action[6]
        if self.v9_cap > 0:
            self.v9_cap -= action[0]
        if self.v10_cap > 0:
            self.v10_cap -= action[0]
        if self.v11_cap > 0:
            self.v11_cap -= action[0]
            self.v11_cap -= action[1]
            self.v11_cap -= action[2]
            self.v11_cap -= action[3]
            self.v11_cap -= action[5]
            self.v11_cap -= action[6]
        if self.v12_cap > 0:
            self.v12_cap -= action[3]
            self.v12_cap -= action[4]
            self.v12_cap -= action[5]
            self.v12_cap -= action[6]
            self.v12_cap -= action[8]
        if self.v13_cap > 0:
            self.v13_cap -= action[5]
            self.v13_cap -= action[7]
            self.v13_cap -= action[8]
        if self.v14_cap > 0:
            self.v14_cap -= action[4]
            self.v14_cap -= action[5]
            self.v14_cap -= action[6]
            self.v14_cap -= action[7]
            self.v14_cap -= action[8]
        if self.v16_cap > 0:
            self.v16_cap -= action[7]
            self.v16_cap -= action[8]
        if self.v17_cap > 0:
            self.v17_cap -= action[7]
        if self.v18_cap > 0:
            self.v18_cap -= action[6]
            self.v18_cap -= action[7]
            self.v18_cap -= action[8]

        return self.get_state()

    def compute_reward(self, action):
        self.reward = 0
        # throughput
        for a in action:
            self.reward += a

        # total_delay
        # delay = 20*action[0] + 21*action[1] + 15*action[2] \
        #     + 12*action[3] + 15*action[4] + 15*action[5]  \
        #     + 18*action[6] + 27*action[7] + 22*action[8]
        # delay = 20*action[0] + 21*action[1] + 15*action[2] + 12*action[3]
        # self.reward = np.exp((-1)*(delay/100))*100
        return self.reward

    def capacity_eff(self):
        # if self.s1_capacity <= 0 and self.d1_capacity <= 0 \
        #         and self.r1_capacity <= 0 and self.r2_capacity <= 0 \
        #         and self.s2_capacity <= 0 and self.d2_capacity <= 0:
        #     return True
        if self.v1_cap <= 0 and self.v16_cap <= 0 \
            and self.v3_cap <= 0 and self.v4_cap <= 0 \
            and self.v5_cap <= 0 and self.v6_cap <= 0 \
            and self.v7_cap <= 0 and self.v17_cap <= 0 \
            and self.v9_cap <= 0 and self.v10_cap <= 0 \
            and self.v11_cap <= 0 and self.v12_cap <= 0 \
            and self.v13_cap <= 0 and self.v14_cap <= 0 \
            and self.v18_cap <= 0:
            return True
        else:
            return False

    def check_termination(self, step_counter):
        # if step_counter > self.num_steps and self.capacity_eff():
        #     return True
        # else:
        #     return False

        if step_counter > self.num_steps or self.capacity_eff():
            return True
        else:
            return False





