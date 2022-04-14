import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    figure, ax = plt.subplots()

    '----------Meeting Room-----------'
    stepLength = ['50', '60', '80', '90', '100']
    QReward = [np.nan, 39.10732167, 19.38239345, 36.75315818, 54.53107293]
    GAReward = [35.44025893, 35.86367859, 36.15181823, 38.01079649, 38.46924296]
    GreedyReward = [63.86823477, 60.94945475, 67.6936825, 72.02001812, 84.69478825]
    A3CReward = [70.33484729, 79.62861346, 100.1454124, 111.9249254, 121.0303124]
    DDPGReward = [44.42488599, 44.78679078, 41.22687269, 44.17998923, 43.96619418]

    '----------Lab-----------'
    # stepLength = ['130', '150', '180', '200']
    # QReward = [44.4950726739395, 55.4180152269206, 92.7396262483108, 101.350845390725]
    # GAReward = [106.5495205, 108.1446887, 135.1763291, 106.0618964]
    # GreedyReward = [161.0486141, 170.956099, 217.0196434, 180.8239676]
    # A3CReward = [157.7788654, 183.4583055, 225.8689418, 253.3521922]
    # DDPGReward = [150.8184818, 137.8646865, 153.3531353, 162.4389439]

    plt.plot(stepLength, QReward, color='c', marker='o', label='Q-Learning')
    plt.plot(stepLength, DDPGReward, color='orange', marker='*', label='PRM-RL')
    plt.plot(stepLength, GAReward, color='b', marker='v', label='GA')
    plt.plot(stepLength, GreedyReward, color='green', marker='x', label='Greedy')
    plt.plot(stepLength, A3CReward, color = 'r', marker = 'p', label = 'A3C')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Exploration Step Length', font2)
    plt.ylabel('Total Reward', font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left')
    plt.savefig('MeetCompareReward.pdf', bbox_inches='tight', dpi=500)
    # plt.ylim(15, 130)
    plt.show()

if __name__ == '__main__':
    main()