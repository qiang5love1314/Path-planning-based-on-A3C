import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from MeetingEnv import *
import pandas as pd
import multiprocessing
import threading
import sys
import time
from scipy.io import loadmat, savemat

N_WORKERS = multiprocessing.cpu_count()

MAX_EP_STEP = 90
MAX_GLOBAL_EP = 100
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.8
ENTROPY_BETA = 0.02 # 稍大
LR_A = 0.15     # learning rate for actor
LR_C = 0.05     # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
alpha = 0.9         # 稍大
global_rate = 0.9   # 稍大

N_S = 2
N_A = 1
A_BOUND = [0,3]

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        self.stateList = np.empty([0, 2], dtype=np.int)
        # self.stateList = pd.DataFrame(columns=4, dtype=np.float)
        self.table = pd.DataFrame(columns = [0,1,2,3])
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4

                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=[0, 1]), A_BOUND[0], A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 50, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 30, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, observation):
        self.create_state_if_not_exists(observation)
        if np.random.uniform() < alpha:
            actions =self.table.loc[observation,:]
            action = self.choose_best_action(actions)
        else:
            action = np.random.choice(4)

        return action

    def choose_best_action(self, actions):
        q_table_cols = self.table.columns
        max_action_value = -sys.maxsize
        max_action_value_list = []

        for idx in range(len(q_table_cols)):
            action_value = actions[idx]
            q_tabl_col = q_table_cols[idx]

            if action_value > max_action_value:
                max_action_value = action_value
                max_action_value_list = [q_tabl_col]
            elif action_value == max_action_value:
                max_action_value_list.append(q_tabl_col)
            else:
                continue

        if len(max_action_value_list) > 1:
            random_action_index = np.random.randint(0, len(max_action_value_list) - 1)
            best_action = max_action_value_list[random_action_index]
        else:
            best_action = max_action_value_list[0]

        return best_action

    def create_state_if_not_exists(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    data=[0] * len([0,1,2,3]),
                    index= [0,1,2,3],
                    name=state
                )
            )

    def check_state_exist(self, state):
        if state not in self.stateList:
            self.stateList = np.append(self.stateList, [(state[0], state[1])], axis=0).tolist()

class Worker(object):
    def __init__(self, name, globalAC):
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.env = Environment()

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        storeAction = []    # 存动作列表
        storeState = []     # 存状态列表
        # aa = loadmat('D:/pythonWork/indoor Location/forth-code/saveModel/action-200_NN100.mat')   # 重新加载复现时再用，仅存动作列表即可
        # actionList = aa['array'][0]

        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                self.env.render()
                a = self.AC.choose_action(str(s))
                # a = actionList[total_step-1]  # 重新加载复现
                s_, r, done = self.env.step(a)
                self.AC.create_state_if_not_exists(str(s_))
                # self.AC.check_state_exist(s_)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                storeAction.append(a)   # 存动作列表
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: np.array(s_).reshape((1,2))})
                        #v_s_ = env.reward[s_[0], s_[1]]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        if s_ != [16,11]:
                            v_s_ = r + GAMMA * v_s_
                        else:
                            v_s_ = r

                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(global_rate * GLOBAL_RUNNING_R[-1] + (1-global_rate) * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

            storeState.append(GLOBAL_RUNNING_R)
        # savemat('D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/reward-90_NN50.mat', {'array': storeState})    # 存奖赏值
        # savemat('D:/pythonWork/indoor Location/forth-code/saveModel/MeetingData/action-90_NN50.mat', {'array': storeAction})   # 存动作列表
        # self.env.final_states()
        # self.env.destroy()

if __name__ == "__main__":
    star_time = time.time()
    SESS = tf.Session()
    env = Environment()

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)

    worker = Worker('ResearchRobot', GLOBAL_AC)
    SESS.run(tf.global_variables_initializer())
    COORD = tf.train.Coordinator()
    COORD.join(worker.work())

    run_time = time.time() - star_time
    # env.final()
    print(run_time)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()