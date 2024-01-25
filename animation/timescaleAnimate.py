#source : youtube video, animating NFL data using timescaleDB
#import all needed modules (TODO: research each module a little)
import configparser
#a module to help us connect to timescale db
# import psycopg2
#to store the data in python, pandas (data frames)
import pandas as pd
import numpy as np
#matplotlib - this is where we can actually visualize the data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as clrs
from matplotlib.animation import FuncAnimation

#get our csv reader ready
import csv
#set up csvList
playerData = open("./smoothData2.csv")
playerDataReader = csv.reader(playerData)
#clear header, store for debugging
header = next(playerDataReader)
#the number of players is (1/2)(x-1), where x is the length of the header
numPlayers = int((.5)*(len(header) -1))
#debug
#print(numPlayers)
#numPlayers is working

#initialize some global variables, we want lists of the x and y data
playerVal = 0
xdatas = []
ydatas = []
for i in range(numPlayers) :
    xdatas.append(0)
    ydatas.append(0)



#create the frisbee field - 110 x 40
def generate_field() :
    #pulled from the code. I understand what all of the variables do except for zorder
    #first coords pair is the upper left corner
    field = patches.Rectangle((0, 0), 110.0, 40.0, linewidth=2,
                            edgecolor='white', facecolor='green', zorder=0)
    #set up display
    #initialize figure and axis data
    fig, ax = plt.subplots(1, figsize=(11, 4))
    ax.add_patch(field)
    #add field lines
    ax.axvline(x=20, color="white", zorder=1)
    ax.axvline(x=90, color="white",zorder=1)
    #add horizontal lines to give axis context
    ax.axhline(y=0.0, color="white",zorder=1)
    ax.axhline(y=40.0, color="white",zorder=1)
    #turn off the axes
    plt.axis('off')
    #debugging
    #plt.show()
    #at this point, the field shows up, working

    #creating scatter plots for the players? Maybe something we want to do
    ax.scatter([], [], c= 'blue', label = 'Cutrules', zorder=2)
    ax.scatter([], [], c= 'red', label = 'Losing team', zorder=2)
    # ax.scatter([], [], c='white' , label = 'Disc', zorder=2)
    #legend creation not working
    ax.legend(loc='upper right')

    

    #we want to return the figure and axis data
    return fig, ax



# plot static graph

fig, ax = generate_field()

#we want a line for each player, marked as a red o
playerList = []
for i in range(numPlayers) :
    ln, = ax.plot([], [], color=(1.0,0.0,0.0), marker='o')
    playerList.append(ln)

#how to plot a single moment
def update(frame):
    nextData = next(playerDataReader)
    
    if (nextData) :
        #update each player
        for i in range (1, len(header), 2) :
            playerVal = int((.5) * (i - 1))
            xdatas[playerVal] = (float(nextData[i]) / 10)
            ydatas[playerVal] = (float(nextData[i+1]) / 10)
            playerList[playerVal].set_data(xdatas[playerVal], ydatas[playerVal])


    # if (nextData) :
    #     xdata = (float(nextData[1]) / 10)
    #     ydata = (float(nextData[2]) / 10)
    # ln.set_data(xdata, ydata)
    return playerList,
#debugging
#plt.show()

anim = FuncAnimation(fig, update, frames=range(1,100), repeat=False, interval=2 )
plt.show()

#notes
#don't think/worry about SQL