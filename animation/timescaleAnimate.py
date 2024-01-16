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
from matplotlib.animation import FuncAnimation

#get our csv reader ready
import csv
#set up csvList
playerData = open("./smoothData1.csv")
playerDataReader = csv.reader(playerData)
#clear header, store for debugging
header = next(playerDataReader)

#initialize some global variables

xdata, ydata = 0.0, 0.0



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
#not really sure why we're plotting it again but here we are
fig, ax = generate_field()
ln, = ax.plot([], [], 'bo')

#how to plot a single moment
def update(frame):
    nextData = next(playerDataReader)
    
    if (nextData) :
        xdata = (float(nextData[1]) / 10)
        ydata = (float(nextData[2]) / 10)
    ln.set_data(xdata, ydata)
    return ln,
#debugging
#plt.show()

anim = FuncAnimation(fig, update, frames=range(1,100), repeat=False )
plt.show()

#notes
#don't think/worry about SQL