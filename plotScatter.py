import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def sampleError(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    return sample

def main():
    figure, ax = plt.subplots()
    '-----Lab-----'
    # sample = sampleError(r'D:\pythonWork\indoor Location\forth-code\Predict-Lab-Error.mat')
    # sample2 = sampleError(r'D:\pythonWork\indoor Location\forth-code\180-Lab-Error.mat')
    # sample3 = sampleError(r'D:\pythonWork\indoor Location\forth-code\150-Lab-Error.mat')
    # sample4 = sampleError(r'D:\pythonWork\indoor Location\forth-code\130-Lab-Error.mat')

    '-----Meeting-----'
    sample = sampleError(r'D:\pythonWork\indoor Location\forth-code\Predict-Meet-Error.mat')
    sample2 = sampleError(r'D:\pythonWork\indoor Location\forth-code\90-Meet-Error.mat')
    sample3 = sampleError(r'D:\pythonWork\indoor Location\forth-code\80-Meet-Error.mat')
    sample4 = sampleError(r'D:\pythonWork\indoor Location\forth-code\60-Meet-Error.mat')

    random = np.random.RandomState(30)
    len1 = random.rand(len(sample))

    ax.scatter(len1, sample, s=120, color = 'r', marker ='o', alpha=0.6, label='Budget: 100')
    ax.scatter(len1, sample2, s=120, color='green', marker='x', alpha=0.6, label='Budget: 90')
    ax.scatter(len1, sample3, s=120, color='c', marker='v', alpha=0.6, label='Budget: 80')
    ax.scatter(len1, sample4, s=120, color='b', marker='p', alpha=0.6, label='Budget: 60')


    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Total Reward',font2)
    plt.ylabel('Localization Error (m)',font2)

    step = np.arange(70,150,8.5)
    ax.set_xticklabels(step)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc = 'upper right')
    figure.savefig('MeetRewardAccuracy.pdf', bbox_inches = 'tight', format='pdf')
    plt.show()

if __name__ == '__main__':
    main()
    pass