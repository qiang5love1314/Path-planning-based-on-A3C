import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    # print(np.mean(sample))
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():
    '-----Lab and Meeting Room-----'
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/Original-Lab-Error.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/Predict-Lab-Error.mat')
    x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/Original-Meet-Error.mat')
    x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/Predict-Meet-Error.mat')

    figure, ax = plt.subplots()
    plt.step(x1, y1, color = 'c', marker ='o', label='Lab-Manual')
    plt.step(x2, y2, color='b', marker='v', label='Lab-Predictive')
    plt.step(x3, y3, color='green', marker='x', label='Meet Room-Manual')
    plt.step(x4, y4, color='r', marker='p', label='Meet Room-Predictive')

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Localization Error (m)',font2)
    plt.ylabel('CDF',font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc = 'lower right')
    # plt.savefig('CDF_PredictAndOriginal_Accuracy.png', bbox_inches = 'tight', dpi=500)
    plt.show()

if __name__ == '__main__':
    main()
    pass