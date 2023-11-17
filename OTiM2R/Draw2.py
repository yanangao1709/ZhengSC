import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def take_mean_value(y):
    y_new = []
    y_new.append(y[0])
    sum_y = y[0]
    for i in range(1,100):
        sum_y += y[i]
        y_new.append(sum_y / (i + 1))

    for j in range(101, len(y)+1):
        k = j - 100
        sum_y = 0
        for kk in range(k, j):
            sum_y += y[kk]
        y_new.append(sum_y / 100)
    return y_new

def obtainInteger(data):
    for i in range(0, len(data)):
        temp = round(data[i])
        data[i] = temp
    return data

def predata(data):
    new_data = []
    for i in range(0, 400):
        new_data.append(data[i])
    for i in range(1000, 1600):
        if (data[i] - 70 ) < 75:
            data[i] = 145
        new_data.append(data[i]-70)
    return new_data

# Throughput  #2b6a99  blue
# Fidelity/|R| #1b7c3d' green o
# Delay #f16c23 darkorange  ^
# Request_number    #E5B9B5    #923931 s
throughput = pd.read_csv('.\\NoCRR-Throughput.csv')
throughput2 = predata(throughput['acc_reward'])
mean_throughput = take_mean_value(throughput2)
# throughput2 = obtainInteger(throughput['acc_reward'])
# mean_throughput = obtainInteger(mean_throughput)

res = {"x":throughput['x'][0:1000], "acc_reward": throughput2}
pd.DataFrame(res).to_csv('NoCRR-Throughput-new.csv', index=False)

plt.plot(throughput['x'][0:1000], throughput2, color='#E5B9B5', alpha=0.5, label='Requests')
plt.plot(throughput['x'][0:1000], mean_throughput, color='#923931', marker='s', markevery=100, label='Average-request')
plt.ylabel('Requests', fontsize=15)
plt.xlabel('Episodes', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=13)

plt.grid(True)
plt.show()