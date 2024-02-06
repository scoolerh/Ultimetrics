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
import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation

# import ffmpeg
plt.rcParams['animation.ffmpeg_path'] ='./animation.mp4'





#get our csv reader ready
import csv
#set up csvList

#NOTE: file format
#file that we're writing in -- format is one column of frame numbers (1, 2, 3, 4, 5), and then 
#pairs of columns representing the number of players. See smoothData1.csv for an example. The first 
#column in a player pair is the x coordinate, the second is the y coordinate. 
playerData = open("./playercoordinates.csv")
#playerData = open("./smoothData2.csv")
playerDataReader = csv.reader(playerData)
#clear header, store for debugging
header = next(playerDataReader)
#the number of players is (1/2)(x-1), where x is the length of the header
numPlayers = int((.5)*(len(header)))
print("num players = " + str(numPlayers))
#keep in mind that the first player is actually the disc

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

#debugging
print(len(xdatas))
print(len(ydatas))



#create the frisbee field - 110 x 40
def generate_field() :
    #pulled from the code. I understand what all of the variables do except for zorder
    #first coords pair is the upper left corner
    field = patches.Rectangle((0, 0), 120.0, 40.0, linewidth=2,
                            edgecolor='white', facecolor='green', zorder=0)
    #set up display
    #initialize figure and axis data
    fig, ax = plt.subplots(1, figsize=(11, 4))
    ax.add_patch(field)
    #add field lines
    ax.axvline(x=25, color="white", zorder=1)
    ax.axvline(x=95, color="white",zorder=1)
    #add horizontal lines to give axis context
    ax.axhline(y=0.0, color="white",zorder=1)
    ax.axhline(y=40.0, color="white",zorder=1)
    #turn off the axes
    plt.axis('off')
    #debugging
    #plt.show()
    #at this point, the field shows up, working

    #creating scatter plots for the players? Maybe something we want to do
    #add the disc
    ax.scatter([], [], c= 'blue', label = 'Cutrules', zorder=2)
    ax.scatter([], [], c= 'red', label = 'Losing team', zorder=2)
    ax.scatter([], [], c= 'purple', label = 'Disc', zorder=2)

    # ax.scatter([], [], c='white' , label = 'Disc', zorder=2)
    #legend creation not working
    ax.legend(loc='upper right')

    

    #we want to return the figure and axis data
    return fig, ax



# plot static graph

fig, ax = generate_field()

#we want a line for each player, marked as a red o
playerList = []
ln, = ax.plot([], [], color=('purple'), marker='o')
playerList.append(ln)
for i in range(1, numPlayers) :
    ln, = ax.plot([], [], color=(1.0,0.0,0.0), marker='o')
    playerList.append(ln)

#how to plot a single moment
def update(frame):
    nextData = next(playerDataReader)
    
    if (nextData) :
        
        #update each player
        #the first set of columns and rows is the disc
        #we will just call the disc "playerVal 1", but it will reference the disc
        for i in range (0, 2) : 
            discVal = 0
            xdatas[discVal] = (float(nextData[i]) / 10)
            ydatas[discVal] = (float(nextData[i+1]) / 10)
            playerList[discVal].set_data(xdatas[discVal], ydatas[discVal])
        for i in range (2, len(header), 2) :
            playerVal = int((.5) * i)
        
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

anim = animation.FuncAnimation(fig, update, frames=range(1,100), repeat=False, interval=100 )

plt.show()
##animation##
FFwriter = animation.FFMpegWriter()
output = open('./animation.mp4', 'w')
anim.save(output, writer = FFwriter)




#notes
#don't think/worry about SQL