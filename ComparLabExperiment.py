#Q-Learning仍比较特殊，需要将Env中的奖赏值置为负数，包括避障与非goal选择动作都置为负数奖赏。-2，-2，-totalReward

import numpy as np
# from LabPathPlanning import *
from GreedyLabSearch import GreedyPredictPath
# from GA_MeetSearch import GAPredictPath
from LabVIF import rawCSI, main
from scipy.io import loadmat, savemat

def QLearningPredict():
    state = loadmat("D:/pythonWork/indoor Location/forth-code/ComparisionExperiments/RL_Q-Learning/LabData/QLab-State100.mat")
    list11 = state['array'][0]

    pathIndex = findIndexQlearning([21,23],list11)[0][0]
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

def findIndex(label, pathPlan):
    index = []
    for i in range(len(pathPlan)):
        index1 = np.where(label[:, 0] == pathPlan[i][0])
        index2 = np.where(label[:, 1] == pathPlan[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index

if __name__ == '__main__':
    predict_list = QLearningPredict()
    # predict_list = GreedyPredictPath()
    # GA实验
    # state = loadmat("D:/pythonWork/indoor Location/forth-code/ComparisionExperiments/Genetic-master/GA_Lab_data/GA-Lab-State200NN200.mat")
    # predict_list = state['array']

    original, label, count = rawCSI()
    indexOfPathPlan = np.array(findIndex(label, predict_list)).flatten()
    diagReward = main()
    sumReward = 0

    for i in range(len(predict_list)):
        signalReward = diagReward[int(predict_list[i][0]), int(predict_list[i][1])]
        sumReward += signalReward

    print(sumReward)