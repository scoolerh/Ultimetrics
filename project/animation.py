import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import csv
import warnings
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
playerData = open("./playercoordinates.csv")
playerDataReader = csv.reader(playerData)
teamData = open("./teams.csv")
teamDataReader = csv.reader(teamData)

header = next(playerDataReader, False)
numPlayers = int((.5)*(len(header)))

#initialize some global variables, we want lists of the x and y data
playerVal = 0
xdatas = []
ydatas = []
for i in range(numPlayers) :
    xdatas.append(0)
    ydatas.append(0)
# will be updated as we go through csv

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
    ax.scatter([], [], c= 'blue', label = 'CUTrules', zorder=2)
    ax.scatter([], [], c= 'red', label = 'Losing Team', zorder=2)
    # ax.scatter([], [], c='white' , label = 'Disc', zorder=2)
    ax.legend(loc='upper right')

    return fig, ax

# plot static graph
fig, ax = generate_field()

#we want a line for each player, marked as a red o
playerList = []
for i in range(numPlayers) :
    team = next(teamData)
    if team == 1: 
        color = '#FF0036'
    else: 
        color = '#467EFF'
    ln, = ax.plot([], [], color=color, marker='o')
    playerList.append(ln)

#how to plot a single moment
def update(frame):
    nextData = next(playerDataReader, False)
    
    if (nextData) :
        #update each player
        for i in range (0, len(nextData)-1, 2) :
            playerVal = int((.5) * (i - 1))
            xdatas[playerVal] = (float(nextData[i+1]))
            ydatas[playerVal] = (float(nextData[i]))
            playerList[playerVal].set_data(xdatas[playerVal], ydatas[playerVal])


    # if (nextData) :
    #     xdata = (float(nextData[1]) / 10)
    #     ydata = (float(nextData[2]) / 10)
    # ln.set_data(xdata, ydata)
    return playerList,

anim = animation.FuncAnimation(fig, update, frames=range(1,numFrames), repeat=False, interval=100)
# anim.save("./animation_moviepy.mp4")
# print('pausing')
writer = animation.FFMpegWriter(
     fps=8, metadata=dict(artist='Conor_And_Taylor'), bitrate=800)
anim.save("frisbeeMovie.mp4", writer=writer)
#plt.show()
##animation##
# FFwriter = animation.FFMpegWriter(fps=10)
# output = open('./animation.mp4', 'w')
# anim.save('./animation.mp4', writer = FFwriter)

# f = "animation.gif"
# writergif = animation.PillowWriter(fps=30) 
# anim.save(f, writer=writergif)

# writervideo = animation.FFMpegWriter(fps=5) 
#print("saving video")
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
# """ 
# f = "animation.gif" 
# writergif = animation.PillowWriter(fps=30, codec='libx264', bitrate=2) 
# anim.save(f, writer=writergif) """

plt.close()
