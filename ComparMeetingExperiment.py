#对比实验包括Q-learning,Greedy,GA。

#其中，Q-learning对Env改动较大，对于非goal的奖赏值需置为负数，很重要，否则找不到解；因为，此环境没有障碍物。
#Q-learning倾向于找到的是最短路径，是不是最优不一定。

#对于Greedy，对Env保持不变，搜索的最终结果并不一定能到达goal，我们要保持尽可能的探索更多的位置以提升总奖赏值。

#对于GA，基于A3C搜索的潜在路径结果进行变化，因此步长、动作不用再次迭代计算；已有路径直接作为染色体效仿郑榕老师做法。
#特别的，GA对于MeetingVIF中奖赏值的返回值仍然是diagReward。由于交叉和变异操作，使得不再是一条连续的路径。

import numpy as np
from MeetingPathPlanning import *
from GreedySearch import GreedyPredictPath
from GA_MeetSearch import GAPredictPath
from MeetingVIF import rawCSI, main
from scipy.io import loadmat, savemat

def QLearningPredict():
    state = loadmat("D:/pythonWork/indoor Location/forth-code/ComparisionExperiments/RL_Q-Learning/QLearning-State100.mat")
    list11 = state['array'][0]

    pathIndex = findIndexQlearning([16,11],list11)[0][0]
    list22 = np.array(list11[int(pathIndex/100)*100 : pathIndex+1]).tolist()
    list.append(list22, [1, 1])
    predict_list = [list(t) for t in set(tuple(xx) for xx in list22)]
    predict_list.sort()
    return predict_list

def findIndexQlearning(label, pathPlan):
    index = []
    pathPlan = np.stack(pathPlan, axis=0)
    index1 = np.where(label[0] == pathPlan[:,0])
    index2 = np.where(label[1] == pathPlan[:,1])
    similar = list(set(index1[0]).intersection(set(index2[0])))
    index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index

def findIndexReward(label, pathPlan):
    index = []
    for i in range(len(pathPlan)):
        index1 = np.where(label[:, 0] == pathPlan[i][0])
        index2 = np.where(label[:, 1] == pathPlan[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index

if __name__ == '__main__':
    # predict_list = QLearningPredict()
    # predict_list = GreedyPredictPath()
    predict_list = GAPredictPath()

    original, label, count = rawCSI()
    indexOfPathPlan = np.array(findIndex(label, predict_list)).flatten()
    diagReward = main()
    sumReward = 0

    for i in range(len(predict_list)):
        signalReward = diagReward[int(predict_list[i][0]), int(predict_list[i][1])]
        sumReward += signalReward

    print(sumReward)