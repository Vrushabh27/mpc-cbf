"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
#x1 and x2 are time series states of agents 1 and 2 respectively
#n is the number of agents
#N is number of iterations for one time horizon
#controller[1] and controller[2] are the Game thereotic MPC controllers for agent 1 and 2 respectively

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpc_cbf import MPC
from plotter import Plotter
import util
import config
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import casadi as ca

# Add number of agents
n=2
N=70 #Number of iteration steps for each agent

controller=[None]*n #Control input intitialization
xf=np.zeros((n,3)) # initialization of final states
final_positions_both_agents=np.zeros((n*N,3)) # initialization of final states for both agents


x1=np.zeros((N,3)) # initialization of times series states of agent 1 
x2=np.zeros((N,3)) # initialization of times series states of agent 1 

xf_minus=np.zeros((n,3))
def main():
    c=0
    Q=config.Q_sp
    R=config.R_sp

    #Scenarios: "doorway" or "intersection"
    scenario="doorway"
    #Add all initial and goal positions of the agents here (Format: [x, y, theta])
    if scenario=="doorway":
        initial=np.array([[-1, 0.5, 0],
                    [-1, -0.5, 0]])
        goals=np.array([[1.5, -0.1, 0],
                    [1.5, 0.1, 0]])
    else:
        initial=np.array([[0.0, -2.0, 0],
                      [-2.0, 0.0, 0]])
        goals=np.array([[0, 1.0, 0],
                    [1.0, 0.0, 0]
                    ])

    if scenario=="doorway":
        ox=1
    else:
        ox=-0.3
        ox1=0.3
    LL=[]
    for j in range(N): # j denotes the j^th step in one time horizon
        for i in range(n):  # i is the i^th agent 

        # Define controller & run simulation for each agent i
            config.x0=initial[i,:]
            config.goal=goals[i,:]
            config.obs=[]
            if scenario=="doorway":
                config.obs= [(ox, 0.3, 0.1),(ox, 0.4, 0.1),(ox, 0.5, 0.1),(ox, 0.6, 0.1),(ox, 0.7, 0.1),(ox, 0.8, 0.1),(ox, 0.9, 0.1),
                  (ox, 1.0, 0.1),(ox, -0.3, 0.1), (ox, -0.3, 0.1),(ox, -0.4, 0.1),(ox, -0.5, 0.1),(ox, -0.6, 0.1),(ox, -0.7, 0.1),      
                (ox, -0.8, 0.1),(ox, -0.9, 0.1)]
                obstacles=config.obs

            else:
                config.obs= [(ox, 0.3, 0.1),(ox, 0.4, 0.1),(ox, 0.5, 0.1),(ox, 0.6, 0.1),(ox, 0.7, 0.1),(ox, 0.8, 0.1),(ox, 0.9, 0.1),
                  (ox, 1.0, 0.1),(ox, -0.3, 0.1), (ox, -0.3, 0.1),(ox, -0.4, 0.1),(ox, -0.5, 0.1),(ox, -0.6, 0.1),(ox, -0.7, 0.1),      
                (ox, -0.8, 0.1),(ox, -0.9, 0.1),

                (ox1, 0.3, 0.1),(ox1, 0.4, 0.1),(ox1, 0.5, 0.1),(ox1, 0.6, 0.1),(ox1, 0.7, 0.1),(ox1, 0.8, 0.1),(ox1, 0.9, 0.1),(ox1, 1.0, 0.1),
                  (ox1, -0.3, 0.1),(ox1, -0.4, 0.1),(ox1, -0.5, 0.1),(ox, -0.6, 0.1),(ox1, -0.7, 0.1),(ox1, -0.8, 0.1),(ox1, -0.9, 0.1),
                  
                  (0.4,ox, 0.1),( 0.5,ox, 0.1),(0.6,ox, 0.1),( 0.7,ox, 0.1),(0.8,ox, 0.1),(0.9,ox, 0.1),(1.0,ox, 0.1),
                  (-0.4,ox, 0.1),(-0.5,ox, 0.1),(-0.6,ox, 0.1),(-0.7,ox, 0.1),(-0.8,ox, 0.1),(-0.9,ox, 0.1),

                ( 0.4,ox1, 0.1),(0.5,ox1, 0.1),(0.6, ox1, 0.1),(0.7,ox1, 0.1),(0.8,ox1, 0.1),(0.9,ox1, 0.1),( 1.0, ox1, 0.1),
                ( -0.4,ox1, 0.1),(-0.5, ox1, 0.1),( -0.6, ox1, 0.1),( -0.7,ox1, 0.1),( -0.8,ox1, 0.1),( -0.9,ox1, 0.1)]
                obstacles=config.obs


            # Setting the maximum velocity limits for both agents
            if i==0:
                config.v_limit=0.3 #For agent 1
            else:
                config.v_limit=0.3  #For agent 2

            # Ensures that other agents act as obstacles to agent i
            for k in range(n):
                if i!=k:
                  config.obs.append((initial[k,0], initial[k,1],0.02)) 

            #Liveliness "on" or "off" can be chosen from here
            liveliness='off'

            #Initialization of MPC controller for the ith agent
            controller[i] = MPC(final_positions_both_agents,j,i,liveliness)

            #final_positions_both_agents stores the final positions of both agents
            #[1,:], [3,:], [5,:]... are final positions of agent 1
            #[2,:], [4,:], [6,:]... are final positions of agent 2
            final_positions_both_agents[c,:]=xf[i,:]

            #The position of agent i is propogated one time horizon ahead using the MPC controller
            xf[i,:]=controller[i].run_simulation_to_get_final_condition(final_positions_both_agents,j,i,liveliness).ravel()

            c=c+1
   
        # Plots
        initial=xf #The final state is assigned to the initial state stack for future MPC

    #x1 and x2 are times series data of positions of agents 1 and 2 respectively
    for ll in range(N-1):
        x1[ll,:]=final_positions_both_agents[n*ll,:]
        x2[ll,:]=final_positions_both_agents[n*ll+1,:]
    

    T=0.4
    L=[]
    # Computation of liveliness value L for agent 1
    for i in range(len(x1)-2):
        vec1=((x1[i+1,0:2]-x1[i,0:2])-(x2[i+1,0:2]-x2[i,0:2]))/T
        vec2=x1[i,0:2]-x2[i,0:2]
        l=(np.arcsin(np.cross(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2))))
        L.append(l)

    #Everything below is plotting

    # Create a figure and axis object for the plot
    fig, ax = plt.subplots()
    liveliness_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Draw the stationary circles
    # circles = [Circle((obstacles[i,0], obstacles[i,1]), obstacles[i,2], fill = False) for i in range(len(obstacles))]
    circles = [Circle((obs[0], obs[1]), obs[2], fill = True) for obs in obstacles]
    if scenario=="doorway":
        rect = patches.Rectangle((ox-0.1,0.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect1 = patches.Rectangle((ox-0.1,-1.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        ax.add_patch(rect)
        ax.add_patch(rect1)
    else:
        length=1
        rect = patches.Rectangle((ox-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect1 = patches.Rectangle((ox1-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect2 = patches.Rectangle((ox1-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect3 = patches.Rectangle((ox-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect4 = patches.Rectangle((-length,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect5 = patches.Rectangle((-length,ox1-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect6 = patches.Rectangle((ox1-0.1,ox1-0.1),1-ox1+0.2,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect7 = patches.Rectangle((ox1,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        ax.add_patch(rect)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)
        ax.add_patch(rect5)
        ax.add_patch(rect6)
        ax.add_patch(rect7)

    # for circle in circles:
    #     ax.add_patch(circle) 
       
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
        if frame<len(x1)-2 and abs(L[frame])< 0.0065:
            liveliness_text.set_text('Liveliness function = ON')
        else:
            liveliness_text.set_text('Liveliness function = OFF')
        return agent1, agent2, liveliness_text,

# Create an animation, use the update function and frames as the length of x1
    ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True)

# Save the animation to a file
    ani.save('agents_animation.mp4', writer='ffmpeg')
    plt.show()



if __name__ == '__main__':
    main()
