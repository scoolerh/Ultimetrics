import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import csv
import warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")

# from matplotlib.animation import FFMpegWriter
#from matplotlib.animation import FuncAnimation
# import ffmpeg

#NOTE: file format
#file that we're writing in -- format is one column of frame numbers (1, 2, 3, 4, 5), and then 
#pairs of columns representing the number of players. The first 
#column in a player pair is the x coordinate, the second is the y coordinate. 

numFrames = 0
with open("./playercoordinates.csv") as f:
    numFrames = sum(1 for line in f)

teamData = open("./teams.csv")
teamDataReader = csv.reader(teamData)
playerDataReader = csv.reader(open("playercoordinates.csv"))
header = next(playerDataReader, False)
print(header)
numPlayers = int((.5)*(len(header)))
#initialize the labels for the players
labels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']


#initialize some global variables, we want lists of the x and y data
playerVal = 0
xdatas = []
ydatas = []
for i in range(numPlayers) :
    xdatas.append(0)
    ydatas.append(0)
# will be updated as we go through csv
def smoothData() :
    ##use the savgol filter to smooth large amounts of data
    #look at time spent, how well the smoothing works
    #we're going to smooth each column individually
    playerDataReader = csv.reader(open("playercoordinates.csv"))
    numColumns = len(next(playerDataReader, False))
    #numColumns works correctly
    #create our 2D array
    data = [[] for x in range(numColumns)]

    #initialize a global variable
    numRows = 0
    for row in playerDataReader :
        #print(row)
        #increment
        
        numRows += 1
        for col in range(len(row)) :
            #print(col)
            data[col].append(float(row[col]))
    #data is correctly initialized

    #create an array of numpy arrays
    numpyData = []
    #create an array of smoothed data
    smoothedData = []
    for col in range(numColumns) :
        
        numpyData.append(np.array(data[col]))
        #these are our parameters to change
        smoothedData.append(savgol_filter(numpyData[col], 10, 3))
    #debug
    # print(data)
    # print(smoothedData)


    #the data at this stage is a list of numpy arrays, which we will need to read index by index until we've backfilled
    #for now we just want to print line by line to see if we'd be adding the correct things
    #initialize the next row
    outfile = open("smoothedplayercoordinates.csv", 'w')
    for row in range(numRows) :
        nextRow = []
        for col in smoothedData :
            #this is cursed but it might work
            if (row < len(col)) :
                nextRow.append(col[row])
        #print(nextRow)
        outfile.write(str(nextRow).strip('[]'))
        outfile.write('\n')
    outfile.close()

#create the frisbee field - 110 x 40
def generate_field() :
    field = patches.Rectangle((0, 0), 110.0, 40.0, linewidth=2, edgecolor='white', facecolor='green', zorder=0)
    #initialize figure and axis data
    fig, ax = plt.subplots(1, figsize=(11, 4))
    ax.add_patch(field)
    #add field lines
    ax.axvline(x=20.0, color="white", zorder=1)
    ax.axvline(x=90.0, color="white",zorder=1)
    #add horizontal lines to give axis context
    ax.axhline(y=0.0, color="white",zorder=1)
    ax.axhline(y=40.0, color="white",zorder=1)
    plt.axis('off')

    #creating scatter plots for the players? Maybe something we want to do
    ax.scatter([], [], c= '#FF0036', label = 'Team 1', zorder=2)
    ax.scatter([], [], c= '#467EFF', label = 'Team 2', zorder=2)
    # ax.scatter([], [], c='white' , label = 'Disc', zorder=2)
    ax.legend(loc='upper right')

    return fig, ax

# plot static graph
fig, ax = generate_field()

#we want a line for each player, marked as a red o or a blue o, depending on their team
playerList = []
for i in range(numPlayers) :
    team = next(teamData)
    if team == "1\n": 
        color = '#FF0036'
    else: 
        color = '#467EFF'
    ln, = ax.plot([], [], color=color, marker='o')
    playerList.append(ln)

#how to plot a single moment
#smooth the data
smoothData()

playerData = open("./smoothedplayercoordinates.csv")
playerDataReader = csv.reader(playerData)
def update(frame):
    nextData = next(playerDataReader, False)
    
    if (nextData) :

        # Remove previous labels
        for text in ax.texts:
            text.remove()

        #update each player
        for i in range (0, len(nextData)-1, 2) :
            playerVal = int((.5) * (i - 1))
            xdatas[playerVal] = (float(nextData[i+1]))
            ydatas[playerVal] = (float(nextData[i]))
            playerList[playerVal].set_data(xdatas[playerVal], ydatas[playerVal])
            # Adding labels for each player
            label = labels[int(i/2)]
            ax.text(xdatas[playerVal], ydatas[playerVal], label, color="black", ha='center', va='center', fontsize=8)
            
            # ax.text(0, 0, label, color="black", ha='center', va='center', fontsize=8)
        # for ln, label in zip(playerList, labels):
        #     ln.set_label(label)
        #     ax.text(0, 0, label, color="black", ha='center', va='center', fontsize=8)

    # if (nextData) :
    #     xdata = (float(nextData[1]) / 10)
    #     ydata = (float(nextData[2]) / 10)
    # ln.set_data(xdata, ydata)
    
    return playerList,

anim = animation.FuncAnimation(fig, update, frames=range(1,numFrames), repeat=False, interval=100)
writer = animation.FFMpegWriter(
     fps=6, metadata=dict(artist='Conor_And_Taylor'), bitrate=800)
anim.save("frisbeeMovie.mp4", writer=writer)
print("Animation complete. ------------------------------------------------------------------------")

plt.close()
