import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def take_mean_value(y):
    y_new = []
    y_new.append(y[0])
    sum_y = y[0]
    for t in range(1, len(y)):
        sum_y += y[t]
        y_new.append(sum_y / (t + 1))
    return y_new


column_name = "acc_reward"
x = [i for i in range(0, 1000)]
data = pd.read_csv(".\\results\\Delay-reward-lr0.1.csv")
reward_1 = data[column_name]
data2 = pd.read_csv(".\\results\\Delay-reward-lr0.01.csv")
reward_2 = data2[column_name]
data3 = pd.read_csv(".\\results\\Delay-reward-lr0.05.csv")
reward_3 = data3[column_name]


plt.plot(x, take_mean_value(reward_1), marker='^', markevery=100, color='#F0C2A2', linestyle='-',
         markerfacecolor='none', label='Learning rate=0.1')
plt.plot(x, take_mean_value(reward_2), marker='<', markevery=100, color='#4182A4', linestyle='-',
         markerfacecolor='none', label='Learning rate=0.01')
plt.plot(x, take_mean_value(reward_3), marker='>', markevery=100, color='#A64036', linestyle='-',
         markerfacecolor='none', label='Learning rate=0.05')

plt.tick_params(labelsize=12)
plt.legend(fontsize=13)  # 让图例生效

plt.gca().yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText = True))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.gca().yaxis.offsetText.set_fontsize(12)

plt.xlabel('Episodes', fontsize=15)  # X轴标签
plt.ylabel("Episode average delay-reward", fontsize=15)  # Y轴标签

plt.grid()
plt.show()
