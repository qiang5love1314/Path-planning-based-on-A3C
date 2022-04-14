import numpy as np
# import tkinter as tk
from mttkinter import mtTkinter as tk
import time
from PIL import Image, ImageTk
from LabVIF import rawCSI, main

# Global variable for dictionary with coordinates for the final path
a = {}
env_height = 21
env_width = 23
pixels = 30
GlobalReward = main()

class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('path planning based-A3C')
        self.geometry('{0}x{1}'.format(env_width * pixels, env_width * pixels))
        self.build_environment()
        l = tk.Label(self, text='The Lab', bg='grey', font=('Arial', 12), width=30, height=2)
        l.pack()

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Key for the dictionaries
        self.i = 0

        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

        # print(self.choose_action([3,6]))

    # Function to build the environment
    def build_environment(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
                                       height= env_height * pixels,
                                       width= env_width * pixels)

        # Creating grid lines
        for row in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_width * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')

        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')

        # Creating objects of  Obstacles
        self.obstacle1 = self.canvas_widget.create_rectangle(0 * 30, 2 * 30, 5 * 30, 5 * 30, fill='black')      #desk1
        self.obstacle2 = self.canvas_widget.create_rectangle(12 * 30, 2 * 30, 21 * 30, 5 * 30, fill='black')    #desk2
        self.obstacle3 = self.canvas_widget.create_rectangle(0 * 30, 8 * 30, 5 * 30, 12 * 30, fill='black')     #desk3
        self.obstacle4 = self.canvas_widget.create_rectangle(7 * 30, 9 * 30, 9 * 30, 11 * 30, fill='black')     #pillar
        self.obstacle5 = self.canvas_widget.create_rectangle(12 * 30, 8 * 30, 21 * 30, 12 * 30, fill='black')   #desk4
        self.obstacle6 = self.canvas_widget.create_rectangle(0 * 30, 15 * 30, 5 * 30, 19 * 30, fill='black')    #desk5
        self.obstacle7 = self.canvas_widget.create_rectangle(10 * 30, 15 * 30, 21 * 30, 19 * 30, fill='black')  #desk6

        # Creating an agent
        self.agent = self.canvas_widget.create_rectangle(0, 0, 30, 30, fill='blue')
        # Final Point
        self.flag = self.canvas_widget.create_rectangle(22 * 30, 20 * 30, 23 * 30, 21 * 30, fill='red')

        self.canvas_widget.pack()

    # Function to reset the environment and start new Episode
    def reset(self):
        self.update()
        time.sleep(0.1)

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_rectangle(0, 0, 30, 30, fill='blue')

        # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        return coordsTransAnget(self.canvas_widget.coords(self.agent))

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = coordsTransAnget(self.canvas_widget.coords(self.agent))
        # move the target, locate at original point
        base_action = np.array([0,0])

        # up
        if action == 0:
            if (state[0]-1)*pixels >= pixels:
                base_action[1] -= pixels
        # down
        elif action == 1:
            if state[0]*pixels <= (env_height-1) * pixels:
                base_action[1] += pixels
        # right
        elif action == 2:
            if state[1]*pixels <= (env_width-1) * pixels:
                base_action[0] += pixels
        # left
        elif action == 3:
            if (state[1]-1)*pixels >= pixels:
                base_action[0] -= pixels

        self.canvas_widget.move(self.agent, base_action[0], base_action[1])
        self.d[self.i] = coordsTransAnget(self.canvas_widget.coords(self.agent))
        next_state = self.d[self.i]

        self.i += 1

        # calculate the reward
        if next_state == coordsTransAnget(self.canvas_widget.coords(self.flag)):
            reward = GlobalReward[21, 23]
            done = True
            # next_state = 'goal'

            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False

        elif next_state in (self.ObstacleList()):
            # reward = -10
            done = True
            self.canvas_widget.move(self.agent, -base_action[0], -base_action[1])   # 后退
            # next_state = coordsTransAnget(self.canvas_widget.coords(self.agent))
            # next_state = 'obstacle'

            Newaction = np.random.permutation(self.n_actions)
            Newaction = list(Newaction)
            Newaction.remove(action)
            stateList = np.empty([0, 2], dtype=np.int)

            for i in range(3):
                next_state = coordsTransAnget(self.canvas_widget.coords(self.agent)) # 返回至前一状态
                if Newaction[i] == 0:
                    next_state[0] -= 1
                elif Newaction[i] == 1:
                    next_state[0] += 1
                elif Newaction[i] == 2:
                    next_state[1] += 1
                elif Newaction[i] == 3:
                    next_state[1] -= 1
                stateList = np.append(stateList, [(next_state[0], next_state[1])], axis=0).tolist()

            ExistState = list(self.d.values())
            find = [x for x in stateList if x not in ExistState]
            if find != []:
                num = len(find)
                num = np.random.choice(num)
                PriorState = find[num]
                reward = 0.5
                # reward = -2
            else:
                num = np.random.choice(3)
                PriorState = stateList[num]
                reward = 0.1
                # reward = -2
            next_state = PriorState

            # clear the dictionary and i
            self.d = {}
            self.i = 0

        else:
            reward = GlobalReward[next_state[0], next_state[1]]
            done = False
            # next_state = 'possible valid path'

        return next_state, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()

    def final(self):
        self.canvas_widget.delete(self.agent)

        self.initial_point = self.canvas_widget.create_oval(15-5, 15-5,
                                                            15+5, 15+5,
                                                            fill='blue',outline='yellow')

        for j in range(len(self.f)):
            print(self.f[j])
            self.track = self.canvas_widget.create_oval(self.f[j][1]+15-5, self.f[j][0]+15-5,
                                                        self.f[j][1]+15+5, self.f[j][0]+15+5,
                                                        fill='blue',outline='yellow')

            a[j] = self.f[j]

    def choose_action(self, observation):
        alpha = 0.8
        action = np.zeros(4)
        if np.random.uniform() < alpha:
            action = np.random.permutation(self.n_actions)

        for i in range(self.n_actions):
            if action[i] == 0:
                observation[0] -= 1
            elif action[i] == 1:
                observation[0] += 1
            elif action[i] == 2:
                observation[1] += 1
            elif action[i] == 3:
                observation[1] -= 1

            if observation not in (self.ObstacleList()):
                return action[i]

    def ObstacleList(self):
        return coordsTransObstacle(self.canvas_widget.coords(self.obstacle1)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle2)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle3)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle4)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle5)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle6)).tolist() + \
               coordsTransObstacle(self.canvas_widget.coords(self.obstacle7)).tolist()

    def final_states(self):
        return a

def coordsTransAnget(agent):
    return [int(agent[3] / pixels), int(agent[2] / pixels)]

def coordsTransObstacle(obstacle):
    goal = [int(obstacle[3] / pixels), int(obstacle[2] / pixels)]
    start = [int(obstacle[1] / pixels) + 1, int(obstacle[0] / pixels) + 1]
    len1 = goal[1] - start[1] + 1
    len2 = goal[0] - start[0] - 1

    obstacleList = np.empty([0, 2], dtype=np.int)
    for i in range(len1):
        target1 = [start[0],start[1] + i]
        obstacleList = np.append(obstacleList, [(target1[0], target1[1])], axis=0)
        target2 = [goal[0], start[1] + i]
        obstacleList = np.append(obstacleList, [(target2[0], target2[1])], axis=0)
    for i in range(len2):
        a = i + 1
        target3 = [start[0]+a,start[1]]
        obstacleList = np.append(obstacleList, [(target3[0], target3[1])], axis=0)
        target4 = [start[0]+a,goal[1]]
        obstacleList = np.append(obstacleList, [(target4[0], target4[1])], axis=0)

    return obstacleList

def initPosition_AgentandFlag():
    original, label = rawCSI()
    CoordList = np.empty((0, 2), dtype=np.int)
    for i in range(len(label)):
        if label[i][0] in [1, 21] or label[i][1] == 23:
            CoordList = np.append(CoordList, [[label[i][0], label[i][1]]], axis=0)
        if label[i][0] in [2, 6, 8, 13, 15, 20] and label[i][1] <= 5:
            CoordList = np.append(CoordList, [[label[i][0], label[i][1]]], axis=0)
        if label[i][1] == 6 and label[i][0] not in [7, 14] or label[i][0] in [7, 14] and label[i][1] == 1:
            CoordList = np.append(CoordList, [[label[i][0], label[i][1]]], axis=0)
    random1 = np.random.randint(len(CoordList), size=2)
    return CoordList[random1[0]],CoordList[random1[1]]

def update():
    for i in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 2
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Environment()
    env.mainloop()
    print(env.step(2))