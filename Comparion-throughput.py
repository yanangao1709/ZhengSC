import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']


column_name = "Total delay"

x=np.arange(5)#柱状图在横坐标上的位置
y1 = [10,11,13,15,16]
y2 = [14,17,19,23,26]
y3 = [5,7,8,10,13]

bar_width=0.2
tick_label = [5, 10, 15, 20, 25]
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}

plt.bar(x, y3, bar_width, color='#006D87', hatch='\\', label='随机策略')
plt.bar(x+bar_width, y2, bar_width, color='#DD6B4F', hatch='+', label='DQN-QN')
plt.bar(x+bar_width*2, y1, bar_width, color='#FFEE6F', hatch='/', label='贪心策略')
# plt.bar(x+bar_width*2, y4, bar_width, color='#535164', hatch='/', label='贪心策略')


plt.legend()
plt.xticks(x,tick_label)

plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Number of requests', fontsize=15)
plt.ylabel("Throughput", fontsize=15)

plt.show()