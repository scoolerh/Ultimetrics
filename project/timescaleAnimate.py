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
from IPython import display 
import os
#from matplotlib.animation import FuncAnimation

# import ffmpeg
plt.rcParams['animation.ffmpeg_path'] ='./animation.mp4'





#get our csv reader ready
import csv
#set up csvList

#NOTE: file format
# input 
#file that we're writing in -- format is one column of frame numbers (1, 2, 3, 4, 5), and then 
#pairs of columns representing the number of players. See smoothData1.csv for an example. The first 
#column in a player pair is the x coordinate, the second is the y coordinate. 

numFrames = 0
with open("./playercoordinates.csv") as f:
    numFrames = sum(1 for line in f)
playerData = open("./playercoordinates.csv")
playerDataReader = csv.reader(playerData)

#clear header, store for debugging
header = next(playerDataReader)
#the number of players is (1/2)(x-1), where x is the length of the header
numPlayers = int((.5)*(len(header)))
print("num players = " + str(numPlayers))
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

# will be updated as we go thorugh csv




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
        for i in range (0, len(header)-1, 2) :
            playerVal = int((.5) * (i - 1))
            xdatas[playerVal] = (float(nextData[i+1]))
            ydatas[playerVal] = (float(nextData[i]))
            playerList[playerVal].set_data(xdatas[playerVal], ydatas[playerVal])


    # if (nextData) :
    #     xdata = (float(nextData[1]) / 10)
    #     ydata = (float(nextData[2]) / 10)
    # ln.set_data(xdata, ydata)
    return playerList,
#debugging
#plt.show()

anim = animation.FuncAnimation(fig, update, frames=range(1,numFrames), repeat=False, interval=100)

plt.show()
##animation##
# FFwriter = animation.FFMpegWriter(fps=10)
# output = open('./animation.mp4', 'w')
# anim.save('./animation.mp4', writer = FFwriter)

# f = "animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# anim.save(f, writer=writergif)

# writervideo = animation.FFMpegWriter(fps=5) 
print("saving video")
# anim.save('animationVideo.mp4', writer=writervideo) 
# plt.close() 
# video = anim.to_html5_video() 
  
# # embedding for the video 

# saving to m4 using ffmpeg writer 
# writervideo = animation.FFMpegWriter(fps=60) 
# output = open('./animation.mp4', 'w')
# anim.save(os.getcwd() + "animation.mp4", writer=writervideo) 
# output.close()
# plt.close() 
# html = display.HTML(video) 

f = "animation.gif" 
writergif = animation.PillowWriter(fps=30, codec='libx264', bitrate=2) 
anim.save(f, writer=writergif)
  
plt.close() 






#notes
#don't think/worry about SQL