"""MPC-CBF controller for a differential drive mobile robot.

Author: Elena Oikonomou
Date: Fall 2022
"""

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from mpc_cbf import MPC
from plotter import Plotter
import util
import config
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Add number of agents
n=2
N=50 #Number of iteration steps for each agent
controller=[None]*n #Just an intitialization
xf=np.zeros((n,3)) # initialization of final states
xff=np.zeros((n*N,3)) # initialization of final states
x1=np.zeros((N,3)) # initialization of final states
x2=np.zeros((N,3)) # initialization of final states
uf=np.zeros((n*N,2))
xf_minus=np.zeros((n,3))
def main():
    c=0
    Q=config.Q_sp
    R=config.R_sp
    #Add all initial conditions of the agents here
    y=np.zeros((1,3))
    x=[np.zeros((0, 3)) for _ in range(n)]
    x=np.array(x) 
    initial=np.array([[-1, 0.5, 0],
                    [-1, -0.5, 0]])
    goals=np.array([[2, 0.1, 0],
                    [2, -0.1, 0]])
    ox=1
    obstacles=np.array([[ox, 0.3, 0.1],
                    [ox, -0.3, 0.1],
                    [ox, 0.4, 0.1],
                    [ox, -0.4, 0.1],
                    [ox, 0.5, 0.1],
                    [ox, -0.5, 0.1],
                    [ox, 0.6, 0.1],
                    [ox, -0.6, 0.1],
                    [ox, 0.7, 0.1],
                    [ox, -0.7, 0.1],
                    [ox, 0.8, 0.1],
                    [ox, -0.8, 0.1],
                    [ox, 0.9, 0.1],
                    [ox, -0.9, 0.1],
                    [ox, 1, 0.1],
                    [ox, -1, 0.1]])
    for j in range(N):
        for i in range(n):         
        # Define controller & run simulation for each agent i
            config.x0=initial[i,:]
            config.goal=goals[i,:]
            config.obs= [(ox, 0.3, 0.1),
                         (ox, 0.4, 0.1),
                (ox, 0.5, 0.1),
                (ox, 0.6, 0.1),
                (ox, 0.7, 0.1),
                (ox, 0.8, 0.1),
                  (ox, 0.9, 0.1),
                  (ox, 1.0, 0.1), 
                  (ox, -0.3, 0.1),
                  (ox, -0.4, 0.1),
                  (ox, -0.5, 0.1),
                  (ox, -0.6, 0.1),
                  (ox, -0.7, 0.1),      
                (ox, -0.8, 0.1),
                (ox, -0.9, 0.1),
                (ox, -1.0, 0.1)]
            if i==0:
                config.v_limit=0.3
            else:
                config.v_limit=0.3
            for k in range(n):
                if i!=k:
                    config.obs.append((initial[k,0], initial[k,1],0.08)) # Ensures that other agents act as obstacles to agent i
            controller[i] = MPC()
            xf_minus[i,:]=controller[i].run_simulation().ravel()
            xf[i,:]=controller[i].run_simulation_to_get_final_condition(xf,xf_minus,0,i).ravel()
            xff[c,:]=xf[i,:]
            c=c+1
        if LA.norm(xf[0,:]-goals[0,:])<0.1 and LA.norm(xf[1,:]-goals[1,:])<0.1 :
            print(j)
            break
            
        print(LA.norm(xf[i,:]-goals[i,:]))
        # Plots
        plotter = Plotter(controller[i])
        initial=xf #The final state is assigned to the initial state stack for future MPC
    # fig, ax = plt.subplots()
    for ll in range(N-1):
        x1[ll,:]=xff[2*ll,:]
        x2[ll,:]=xff[2*ll+1,:]
    # print(x1)
    # print(x2)
    T=0.4
    L=[]
    for i in range(len(x1)-2):
        vec1=((x1[i+1,0:2]-x1[i,0:2])-(x2[i+1,0:2]-x2[i,0:2]))/T
        vec2=x1[i,0:2]-x2[i,0:2]
        l=abs(np.arccos(np.dot(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2))))
        L.append(l)
    print(L)
    # Create a figure and axis object for the plot
    fig, ax = plt.subplots()
    liveliness_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # Draw the stationary circles
    circles = [Circle((obstacles[i,0], obstacles[i,1]), obstacles[i,2], fill = False) for i in range(len(obstacles))]
    for circle in circles:
        ax.add_patch(circle)    
# Initialize two empty plots for agent 1 and agent 2
    agent1, = plt.plot([], [], 'ro', markersize=5)
    agent2, = plt.plot([], [], 'bo', markersize=5)

# Function to initialize the plots, returns the plot objects
    def init():
        ax.set_xlim(-2, 4)
        ax.set_ylim(-2, 4)
        liveliness_text.set_text('Liveliness function = OFF')

        return agent1, agent2,liveliness_text,

# Function to update the plots, returns the plot objects
    def update(frame):
        agent1.set_data(x1[frame, 0], x1[frame, 1])
        agent2.set_data(x2[frame, 0], x2[frame, 1])
        # Update the liveliness text based on the value of k
        if frame<len(x1)-2 and L[frame]< 0.3:
            liveliness_text.set_text('Liveliness function = ON')
        else:
            liveliness_text.set_text('Liveliness function = OFF')
        return agent1, agent2, liveliness_text,

# Create an animation, use the update function and frames as the length of x1
    ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True)

# Save the animation to a file
    ani.save('agents_animation.mp4', writer='ffmpeg')

    plt.show()






    # plt.show()

        # Store results
    util.save_mpc_results(controller[i])


if __name__ == '__main__':
    main()
