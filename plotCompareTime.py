import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    figure, ax = plt.subplots()

    '----------Meeting Room-----------'
    # stepLength = ['50', '60', '80', '90', '100']
    # QTime = [91.17277122, 100.7034054, 115.8698912, 143.4465036, 152.3088105]
    # GATime = [7.268719196, 7.68108201, 7.252644777, 7.56436038, 7.238329887]
    # GreedyTime = [0.597355843, 0.769938231, 0.943301201, 1.047406197, 1.180099487]
    # A3CTime = [82.42205191, 97.651896, 128.9162133, 138.1681538, 151.726048]
    # DDPGTime = [96.7571249, 118.6409781, 158.9223232, 167.8978231, 184.7890091]

    '----------Lab-----------'
    stepLength = ['130', '150', '180', '200']
    QTime = [190.0392919, 241.1754351, 260.097682, 317.3519268]
    GATime = [12.5918323993682, 14.6539039611816, 12.47384881973268, 15.78796029]
    GreedyTime = [ 1.940280676, 2.240297556, 2.515559912, 2.783963919]
    A3CTime = [ 186.7915368, 225.0200441, 272.6645026, 300.1068244]
    DDPGTime = [244.958087, 288.222383, 340.3679209, 384.8758578]

    plt.plot(stepLength, QTime, color='c', marker='o', label='Q-Learning')
    plt.plot(stepLength, DDPGTime, color='orange', marker='*', label='PRM-RL')
    plt.plot(stepLength, GATime, color='b', marker='v', label='GA')
    plt.plot(stepLength, GreedyTime, color='green', marker='x', label='Greedy')
    plt.plot(stepLength, A3CTime, color = 'r', marker = 'p', label = 'A3C')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Exploration Step Length', font2)
    plt.ylabel('Run Time (s)', font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc='upper left')
    # plt.savefig('LabCompareTime.pdf', bbox_inches='tight', dpi=500)
    # plt.ylim(15, 100)
    plt.show()

if __name__ == '__main__':
    main()