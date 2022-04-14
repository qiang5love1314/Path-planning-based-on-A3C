import os
from scipy.io import loadmat
import numpy as np
import math
from matplotlib import pyplot as plt
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
import GPy
# import networkx as nx

def main():
    original, label, count = rawCSI()
    originalData = np.array(original[:, 0:3*30*1000], dtype='float')
    originalData = SimpleImputer(copy=False).fit_transform(originalData)

    '''-------原始数据高斯回归处理-------'''
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(originalData, label)
    # result = gpr.predict(originalData, return_std=False)  #预测什么数据待定

    from sklearn.manifold import MDS
    pointRowGaussSum = get_gaussian(originalData)   #.sum(axis=1)
    normalGauss = pointRowGaussSum / sum(pointRowGaussSum)
    betweenness = MDS(2, random_state=10).fit_transform(normalGauss)
    # plt.scatter(betweenness[:, 0], betweenness[:, 1])
    # plt.scatter(np.arange(0, 317, 1), betweenness[:, 0])
    # plt.show()

    '''-----模仿郑榕老师强化学习建模部分-----'''
    dimenReduce = MDS(2,random_state=10).fit_transform(originalData)    #数据降维

    kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)   #高斯核函数，数据采集过程符合独立假设，本质上不知道数据的空间分布
    size = len(label)
    mu = np.mean(label, axis=1)
    cov = kernelRBF.K(label, label)         #假设认定数据符合(多元)高斯分布，并拟合
    y_sim = np.random.multivariate_normal(mu, cov, size=size)
    H_y = ComputeDifferentialEntropy(cov, size)     #计算假定情况下所有采样点的微分熵

    traindata, testdata, trainlabel, testlabel = train_test_split(dimenReduce, label, test_size=0.9, random_state=20)
    model = GPy.models.GPRegression(traindata, trainlabel, kernel=kernelRBF)    #计算超参数
    model.optimize()    #由真实数据去修正假定的多元高斯分布
    # model.plot()
    # model.posterior_samples_f()
    # print(model.predict(testlabel))

    gaussian_variance = model.param_array[2]

    part1 = kernelRBF.K(label, trainlabel)
    part2 = np.linalg.inv(kernelRBF.K(trainlabel, trainlabel) + math.pow(gaussian_variance, 2) * np.eye(len(trainlabel)))
    part3 = kernelRBF.K(trainlabel, label)
    covPlus= cov - np.dot(np.dot(part1, part2), part3)      #由少量采集数据计算微分熵
    H_yAnds = ComputeDifferentialEntropy(covPlus, size)

    'compute reward based MI'
    reward_MI = H_y - H_yAnds   # the kernel function usually has some hyper-parameters which may not be known in advance ?
    zeroReward = SwapValue(np.diag(reward_MI))
    xLabel = getXlabel()
    yLabel = getYlabel()
    count = 0
    diagReward = np.zeros((17, 12), dtype=np.float)
    for i in range(16):
        for j in range(11):
            filePath = r"/Users/zhuxiaoqiang/Downloads/55SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                swap = zeroReward[count]
                diagReward[int(xLabel[i]), int(yLabel[j])] = swap
                count += 1


    # print(reward_MI)

    # G = nx.Graph()
    # nodes = list(range(size))
    # G.add_nodes_from(nodes)
    #
    # coordinates = label
    # vnode = np.array(coordinates)
    # npos = dict(zip(nodes, vnode))
    # nlabels = dict(zip(nodes, nodes))
    #
    # edges = []
    # for idx in range(size - 1):
    #     edges.append((idx, idx + 1))
    # edges.append((size - 1, 0))
    # G.add_edges_from(edges)
    #
    # nx.draw_networkx_nodes(G, npos, node_size=80, node_color="#6CB6FF")  # 绘制节点
    # nx.draw_networkx_edges(G, npos, edges)  # 绘制边
    # nx.draw_networkx_labels(G, npos, nlabels)  # 标签
    #
    # plt.show()

    '''--------多元高斯回归拟合CSI分布（仅由坐标来预测）----------'''
    # fig = plt.figure(figsize=(12, 8))     #betweenness换成y_sim
    # ax = Axes3D(fig)
    # X, Y = np.meshgrid(label[:,0], label[:,1])
    # ax.plot_surface(X, Y, betweenness, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # # plt.scatter(X, Y, betweenness)      #该拟合结果是否正确，待定
    # plt.show()

    '''--------原始指纹库条件数画图----------'''
    # plotNormCond(originalData, label)
    # plotBarCond(originalData, label)

    return diagReward

def SwapValue(x):         #归一化
    max = np.max(x)
    min = np.min(x)
    k = (5-1) / (max-min)
    value = k *(x-min)+1
    return value

def getXlabel():
        xLabel = []
        for i in range(16):     #横坐标
            str = '%d' % (i + 1)
            xLabel.append(str)
        return xLabel

def getYlabel():
        yLabel = []
        for j in range(11):     #纵坐标
            if(j<9):
                num=0
                str= '%d%d' % (num, j+1)
                yLabel.append(str)
            else:
                yLabel.append('%d'% (j+1) )
        return yLabel

def rawCSI():
        xLabel = getXlabel()
        yLabel = getYlabel()
        count = 0
        originalCSI=np.zeros((176, 135000), dtype=np.float)
        newName = []
        label = np.empty((0, 2), dtype=np.int)

        for i in range(16):
            for j in range(11):
                filePath = r"/Users/zhuxiaoqiang/Downloads/55SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
                name = xLabel[i] + yLabel[j]
                if (os.path.isfile(filePath)):
                    c = loadmat(filePath)
                    CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                    originalCSI[count, :] = CSI
                    newName.append(name)
                    label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                    count += 1
        return originalCSI, label, count

def ComputeDifferentialEntropy(cov, size):
        hyper_parameter = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                hyper_parameter[i, j] = 0.5 * math.log(abs(cov[i, j]), math.e) + size / 2.0 * \
                                        (1 + math.log(2 * math.pi, math.e))
        return hyper_parameter

def get_gaussian(values):           #高斯分布
        mu = np.mean(values)
        sigma = np.std(values)
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
            (np.power(np.e, -(np.power((values - mu), 2) / (2 * np.power(sigma, 2)))))
        return y

def normfun(x,mu,sigma):
        pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return pdf

def configOfPicture():
        figure, ax = plt.subplots()
        plt.tick_params(labelsize=15)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

def getArrayOfCond(originalData, label):
        arrayOfCond = []
        random = np.random.RandomState(10)
        for i in range(100):
            random1 = random.randint(len(label), size=2)
            A = np.vstack((originalData[random1[0]], originalData[random1[1]]))
            condNum = np.linalg.cond(A)
            arrayOfCond.append(condNum)
        return arrayOfCond

def plotNormCond(originalData, label):  # 数据相似度——正太分布
        arrayOfCond = getArrayOfCond(originalData, label)
        mean = np.mean(arrayOfCond)
        std = np.std(arrayOfCond)
        x = np.arange(0, 20, 0.1)
        y = normfun(x, mean, std)

        configOfPicture()
        plt.plot(x,y,'r')
        plt.hist(arrayOfCond, bins=35, rwidth=0.9, normed=True, color = 'steelblue', edgecolor = 'black')
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 15,
                 }
        plt.ylabel('Percentage', font2)
        plt.xlabel('Condition Number', font2)
        # plt.savefig('NormCond.png', bbox_inches = 'tight', dpi=500)
        plt.show(dpi=500)

def plotBarCond(originalData, label):   # 条件数分布直方图
    arrayOfCond = getArrayOfCond(originalData, label)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    configOfPicture()
    plt.bar(np.arange(0,100,1),arrayOfCond, color='steelblue', edgecolor = 'black')
    plt.xlabel('Index', font2)
    plt.ylabel('Condition Number', font2)
    # plt.savefig('BarCond.png', bbox_inches = 'tight', dpi=500)
    plt.show(dpi=500)

if __name__ == '__main__':
    test = main()

    print(test)