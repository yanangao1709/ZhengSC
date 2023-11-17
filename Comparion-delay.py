import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']


column_name = "Total delay"

x=np.arange(5)#柱状图在横坐标上的位置
y1 = [14.22,16.34,18.69,21.76,26.66]
y2 = [9.23,11.56,14.3,18.22,23.78]
y3 = [17.22,19.34,21.69,24.76,29.66]

bar_width=0.2
tick_label = [5, 10, 15, 20, 25]
#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}

plt.bar(x, y3, bar_width, color='#4182A4', hatch='\\', label='随机策略')
plt.bar(x+bar_width, y2, bar_width, color='#A64036', hatch='+', label='DQN-QN')
plt.bar(x+bar_width*2, y1, bar_width, color='#F0C2A2', hatch='/', label='贪心策略')
# plt.bar(x+bar_width*2, y1, bar_width, color='#354E6B', hatch='/', label='贪心策略')

plt.legend()
plt.xticks(x,tick_label)

plt.tick_params(labelsize=12)
plt.legend(fontsize=13)
plt.xlabel('Number of requests', fontsize=15)
plt.ylabel("Total delay", fontsize=15)

plt.show()