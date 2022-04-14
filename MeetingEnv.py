import numpy as np
# import tkinter as tk
from mttkinter import mtTkinter as tk
import time
from PIL import Image, ImageTk
from MeetingVIF import rawCSI, main

# Global variable for dictionary with coordinates for the final path
a = {}
env_height = 16
env_width = 11
pixels = 30
GlobalReward = main()

class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('path planning based-A3C')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))
        self.build_environment()
        l = tk.Label(self, text='The Meeting', bg='grey', font=('Arial', 12), width=30, height=2)
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
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')

        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='black')


        # Creating an agent
        self.agent = self.canvas_widget.create_rectangle(0, 0, 30, 30, fill='blue')
        # Final Point
        self.flag = self.canvas_widget.create_rectangle(10 * 30, 15 * 30, 11 * 30, 16 * 30, fill='red')

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
            reward = GlobalReward[16, 11]
            done = True
            # next_state = 'goal'

            # 118-138后加的
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

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

    def final_list(self):
        for j in range(len(self.f)):
            a[j]=self.f[j]
        return a

def coordsTransAnget(agent):
    return [int(agent[3] / pixels), int(agent[2] / pixels)]

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