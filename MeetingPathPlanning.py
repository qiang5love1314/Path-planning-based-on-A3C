from scipy.io import loadmat, savemat
global GLOBAL_EP
from MeetingEnv import *
from ComparMeetingExperiment import *
import numpy as np
from MeetingVIF import rawCSI
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.manifold import MDS
import GPy
from GreedySearch import GreedyPredictPath
from GA_MeetSearch import GAPredictPath
from DDPG_MeetSearch import DDPGPredictPath

MAX_EP = 100    # 迭代次数
Max_step = 100   # 步长

actionFile = "action-100_NN50.mat"
stateFile = "state-100_NN50.mat"
rewardFile = "reward-100_NN50.mat"

def main22():
    # pathPlan, maxReward = OptimalPath(rewardFile)
    # print(len(pathPlan))
    # storeStateList(actionFile)

    '''对比实验结果'''
    # pathPlan = QLearningPredict()
    # pathPlan = GreedyPredictPath()
    # pathPlan = GAPredictPath()
    pathPlan = DDPGPredictPath()

    original, label, count = rawCSI()
    originalData = np.array(original[:, 0 : 3 * 30 * 1000], dtype='float')
    originalData = SimpleImputer(copy=False).fit_transform(originalData)

    # 获取最佳路径的索引及CSI数据包
    indexOfPathPlan = np.array(findIndex(label, pathPlan)).flatten()
    CSIofPathPlan = getCSIdata(originalData, indexOfPathPlan)
    labelofPathPlan = np.array([label[x] for x in indexOfPathPlan])

    # 获取先验数据的索引及CSI数据包
    traindata, testdata, trainlabel, testlabel = train_test_split(originalData, label, test_size=0.9, random_state=20)
    indexOfPilotCSI = np.array(np.sort(findIndex(label, trainlabel), axis=0)).flatten()
    CSIofPilot = getCSIdata(originalData, indexOfPilotCSI)
    labelofPilot = np.array([label[x] for x in indexOfPilotCSI])

    # 两次采集数据拼接
    indexOfTwoCSI = np.array(np.unique(np.hstack((indexOfPathPlan, indexOfPilotCSI)))).flatten()
    CSIofTwo = getCSIdata(originalData, indexOfTwoCSI)
    labelofTwoCSI = np.array([label[x] for x in indexOfTwoCSI])

    # 其余位置点坐标
    indexOfPredict = list(set(list(range(len(label)))).difference(set(indexOfTwoCSI)))
    labelofPredict = np.array([label[x] for x in indexOfPredict])

    # CSIofTwo = MDS(10, random_state=10).fit_transform(CSIofTwo)
    # originalData = MDS(10, random_state=10).fit_transform(originalData)
    # pls2 = PLSRegression(n_components=2)
    # pls2.fit(labelofTwoCSI, CSIofTwo)
    # yy = pls2.predict(labelofPredict)
    # print(yy)

    # 预测其余位置CSI指纹的分布
    kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)
    cov = kernelRBF.K(label, label)
    mu = np.mean(labelofTwoCSI, axis=1)
    model = GPy.models.GPRegression(labelofTwoCSI, CSIofTwo)  # 计算超参数
    model.optimize()
    # y = model.predict(labelofPredict)

    sum = np.zeros((len(indexOfPredict), 3*30*1000), dtype=np.float)
    for j in range(len(indexOfPredict)):
        Temp = 0
        for i in range(len(indexOfTwoCSI)):
            temp = cov[indexOfTwoCSI[i],indexOfPredict[j]]*CSIofTwo[i]
            Temp += temp
        sum[j,:] = Temp

    PredictFingerprint = np.zeros((176, 3*30*1000), dtype=np.float)
    for i in range(176):
        if i in indexOfTwoCSI:
            index1 = list(indexOfTwoCSI).index(i)
            PredictFingerprint[i,:] = CSIofTwo[index1]
        elif i in indexOfPredict:
            index2 = list(indexOfPredict).index(i)
            PredictFingerprint[i,:] = sum[index2]

    # n = 5
    predictCSI = abs(Zscorenormalization(PredictFingerprint))
    originalCSI = abs(Zscorenormalization(originalData))

    traindata1, testdata1, trainlabel1, testlabel1 = train_test_split(predictCSI, label, test_size=0.2, random_state=20)
    from sklearn.neighbors import KNeighborsRegressor
    KNN = KNeighborsRegressor(n_neighbors=5).fit(traindata1, trainlabel1)
    prediction = KNN.predict(testdata1)
    print(accuracyPre(prediction, testlabel1), 'm')
    print(accuracyStd(prediction, testlabel1), 'm')
    # saveTestErrorMat(prediction, testlabel1, 'Original-Meet-Error')

def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def getCSIdata(originalData, indexOfPathPlan):
    CSI = []
    for i in range(len(indexOfPathPlan)):
        CSI.append(originalData[indexOfPathPlan[i]])
    # label[indexOfPathPlan[i]] 获取标签
    CSI = np.array(CSI)
    return CSI

def findIndex(label, pathPlan):
    index = []
    for i in range(len(pathPlan)):
        index1 = np.where(label[:, 0] == pathPlan[i][0])
        index2 = np.where(label[:, 1] == pathPlan[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    index = [x for x in index if x]  # 删除空元素
    return index

def OptimalPath(rewardFile):
    possiblePath, stateLabel = findPossiblePath(stateFile)
    reward = loadmat("/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/saveModel/MeetingData" + rewardFile)
    rewardList = reward['array'][0]
    numOfpath = len(stateLabel)
    valueOfReward = []
    for i in range(numOfpath):
        valueOfReward.append(rewardList[stateLabel[i]])
    max_index = np.argmax(np.array(valueOfReward))
    OptimalPath = possiblePath[int(max_index)]
    return OptimalPath, np.max(valueOfReward)

def storeStateList(actionFile):
    total = 1
    action = loadmat("/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/saveModel/MeetingData" + actionFile)
    actionList = action['array'][0]
    stateList = []
    env = Environment()
    GLOBAL_EP = 0
    while GLOBAL_EP < MAX_EP:
        env.reset()
        for ep_t in range(Max_step):
            env.render()
            a = actionList[total - 1]  # 重新加载复现
            s_, r, done = env.step(a)
            stateList.append(s_)
            done = True if ep_t == Max_step - 1 else False
            total += 1
            if done:
                GLOBAL_EP += 1
        print(GLOBAL_EP)
    env.destroy()
    # savemat('D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/state-90_NN50.mat', {'array': stateList})

def findPossiblePath(stateFile):
    possiblePath = []
    stateLabel = []
    state = loadmat("/Users/zhuxiaoqiang/Desktop/IEEE Trans/Fourth paper/代码/forth-code/saveModel/MeetingData"+ stateFile)
    stateList = np.reshape(state['array'], (MAX_EP, Max_step , 2))
    for i in range(MAX_EP):
        a = np.array(stateList[i]).tolist()
        list.append(a, [1,1])
        new_list = [list(t) for t in set(tuple(xx) for xx in a)]
        new_list.sort()
        if [1,1] and [16,11] in new_list:
            possiblePath.append(new_list)
            stateLabel.append(i)
    return possiblePath, stateLabel
    #possiblePath[i] 可以遍历所有潜在的结果路径  200-NN200有6条 在第[13, 25, 53, 79, 86, 98]次迭代

def accuracyPre(predictions, labels):
    return  np.mean(np.sqrt(np.sum((predictions-labels)**2,1))) * 60 / 100

def accuracyStd(predictions , testlabel):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 60 / 100
    return np.std(sample)

def saveTestErrorMat(predictions, testlabel, fileName):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 60 / 100
    savemat(fileName+'.mat', {'array': sample})

if __name__ == '__main__':
    main22()
    pass