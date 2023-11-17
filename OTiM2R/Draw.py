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

def draw_noCRR():
    throughput = pd.read_csv('.\\Throughput.csv')
    mean_throughput = take_mean_value(throughput['acc_reward'])

    throughput_noCRR = pd.read_csv('.\\NoCRR-Throughput-new.csv')
    mean_throughput_noCRR = take_mean_value(throughput_noCRR['acc_reward'])
    thr = []
    for i in range(0, len(mean_throughput_noCRR)):
        thr.append(throughput_noCRR['acc_reward'][i] - 12)
    mean_thr = take_mean_value(thr)

    plt.plot(throughput['x'], throughput['acc_reward'], color='#2b6a99', alpha=0.5, label='')
    plt.plot(throughput['x'], mean_throughput, color='blue', marker='*', markevery=100, label='OTOH')
    plt.plot(throughput_noCRR['x'], thr, color='#7F7F7F', alpha=0.5, label='')
    plt.plot(throughput_noCRR['x'], mean_thr, color='#262626', marker='v', markevery=100, label='OTOH_NoCRR')
    plt.plot(565, 142, color='#C00000', marker='D', markersize=8)

    plt.ylabel('Throughput', fontsize=15)
    plt.xlabel('Episodes', fontsize=15)


    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def draw_requestnumber():
    x = [20,40,60,80,100]
    y = [210, 220, 229, 237, 245]
    y_NoCRR = [148, 158, 166, 175, 182]

    plt.plot(x, y, color='blue', linestyle='dashed', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.plot(x, y_NoCRR, color='#262626', linestyle='dashdot', marker='v', markevery=1, markersize=10, label='OTOH_NoCRR')
    plt.ylabel('Optimal-throughput', fontsize=15)
    plt.xlabel('Request', fontsize=15)
    plt.xticks([20,40,60,80,100])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def balance_point():
    x = [20, 40, 60, 80, 100]
    ba_p = [570, 550, 520, 480, 420]
    plt.plot(x, ba_p, color='red', marker='s', markevery=1, markersize=10, label='Balance point')
    plt.ylabel('Episodes', fontsize=15)
    plt.xlabel('Request', fontsize=15)
    plt.xticks([20, 40, 60, 80, 100])
    plt.yticks([420, 460, 500, 540, 580])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def node_capacity():
    x = [10, 12, 14, 16, 18]
    y = [15, 15, 16, 18, 20]
    y_NoCRR = [14, 14, 15, 15, 16]
    plt.plot(x, y, color='blue', linestyle='dashed', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.plot(x, y_NoCRR, color='#262626', linestyle='dashdot', marker='v', markevery=1, markersize=10, label='OTOH_NoCRR')
    plt.ylabel('Success request', fontsize=15)
    plt.xlabel('Node capacity', fontsize=15)
    plt.xticks([10, 12, 14, 16, 18])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def draw_node_capacity2():
    x = [10, 12, 14, 16, 18]
    yQ_CAST = [8, 9, 9, 11, 13]
    yREPS = [11, 11,12,13, 14]
    yEFiRAP = [12, 12, 14,14, 15]
    yQ_LEAP = [12, 12, 14,14, 15]
    ySEE = [13, 13, 14, 16, 17]
    yMulti_R = [14, 15, 15, 17, 18]
    yOTiM2R = [15, 15, 16, 18, 20]
    # plt.plot(x, yQ_CAST, color='#206E9E', linestyle='-', marker='p', markevery=1, markersize=10, label='Q-CAST')
    # plt.plot(x, yREPS, color='#262626', linestyle='-', marker='v', markevery=1, markersize=10,label='REPS')
    # plt.plot(x, yEFiRAP, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='EFiRAP')
    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Success request', fontsize=15)
    plt.xlabel('Node capacity', fontsize=15)
    plt.xticks([10, 12, 14, 16, 18])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def draw_requestnum_all():
    x = [20, 40, 60, 80, 100]
    yQ_CAST = [150, 159, 171, 185, 192]
    yREPS = [165, 176, 179, 187, 199]
    yEFiRAP = [175, 188.9, 197, 207, 216]
    yQ_LEAP = [175, 188.9, 197, 207, 216]
    ySEE = [180, 203, 212, 222, 225]
    yMulti_R = [190, 205.6, 218.2, 225, 229.9]
    yOTiM2R = [210, 220, 229, 237, 245]

    # plt.plot(x, yQ_CAST, color='#206E9E', linestyle='-', marker='p', markevery=1, markersize=10, label='Q-CAST')
    # plt.plot(x, yREPS, color='#262626', linestyle='-', marker='v', markevery=1, markersize=10, label='REPS')
    # plt.plot(x, yEFiRAP, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='EFiRAP')
    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Optimal-throughput', fontsize=15)
    plt.xlabel('Request number', fontsize=15)
    plt.xticks([20, 40, 60, 80, 100])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def throughput_fidelity():
    x = [140, 160, 180, 200, 220, 240, 260]
    yQ_CAST = [0.42, 0.48, 0.51, 0.55, 0.59,0.63,0.67]
    yREPS = [0.42, 0.48, 0.53, 0.58,0.65,0.61,0.69]
    yEFiRAP = []
    yQ_LEAP = [0.45, 0.50, 0.52, 0.58, 0.67, 0.69, 0.74]
    ySEE = [0.45, 0.49, 0.53, 0.59, 0.67, 0.72, 0.77]
    yMulti_R = [0.50, 0.57, 0.63, 0.69, 0.72, 0.79, 0.85]
    yOTiM2R = [0.58, 0.63, 0.72, 0.80, 0.85, 0.92, 0.98]

    # plt.plot(x, yQ_CAST, color='#206E9E', linestyle='-', marker='p', markevery=1, markersize=10, label='Q-CAST')
    # plt.plot(x, yREPS, color='#262626', linestyle='-', marker='v', markevery=1, markersize=10,label='REPS')
    # plt.plot(x, yEFiRAP, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='EFiRAP')
    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Fidelity/|R|', fontsize=15)
    plt.xlabel('Throughput', fontsize=15)
    plt.xticks([140, 160, 180, 200, 220, 240, 260])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def throughput_delay():
    x = [140, 160, 180, 200, 220, 240, 260]
    yQ_CAST = [23.78, 23.023, 22.456, 22.02, 21.567, 21.03, 20.56]
    yREPS = [23.111, 22.56, 22.10, 21.67, 20.345, 19.678, 19.023]
    yEFiRAP = []
    yQ_LEAP = [21, 20.56, 19.456, 19.012, 18.56, 18.1, 17.99]
    ySEE = [21,22, 20.87, 20.81, 19.456, 19.012, 18.112]
    yMulti_R = [20.2987, 19.366, 18.501, 18.01, 17.2, 16.12, 15.23]
    yOTiM2R = [18, 17.2, 15.9, 15.14, 14.789, 14.0987,13.678]

    # plt.plot(x, yQ_CAST, color='#206E9E', linestyle='-', marker='p', markevery=1, markersize=10, label='Q-CAST')
    # plt.plot(x, yREPS, color='#262626', linestyle='-', marker='v', markevery=1, markersize=10,label='REPS')
    # plt.plot(x, yEFiRAP, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='EFiRAP')
    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Delay/|R|', fontsize=15)
    plt.xlabel('Throughput', fontsize=15)
    plt.xticks([140, 160, 180, 200, 220, 240, 260])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # throughput_delay()
    # throughput_fidelity()
    # draw_requestnum_all()
    # draw_node_capacity2()
    node_capacity()
    # balance_point()
    # draw_requestnumber()
    # draw_noCRR()

    # Throughput  #2b6a99  blue *
    # Fidelity/|R| #1b7c3d' green o
    # Delay #f16c23 darkorange  ^
    # Request_number    #E5B9B5    #923931 s
    # throughput = pd.read_csv('.\\NoCRR-Throughput.csv')
    # mean_throughput = take_mean_value(throughput['acc_reward'])
    # # throughput2 = obtainInteger(throughput['acc_reward'])
    # # mean_throughput = obtainInteger(mean_throughput)
    #
    # plt.plot(throughput['x'], throughput['acc_reward'], color='#E5B9B5', alpha=0.5, label='Requests')
    # plt.plot(throughput['x'], mean_throughput, color='#923931', marker='s', markevery=100, label='Average-request')
    # plt.ylabel('Requests', fontsize=15)
    # plt.xlabel('Episodes', fontsize=15)
    # plt.tick_params(labelsize=12)
    # plt.legend(fontsize=13)
    #
    # plt.grid(True)
    # plt.show()