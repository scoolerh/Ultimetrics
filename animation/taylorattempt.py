import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib as mpl 

mpl.rcParams['animation.ffmpeg_path'] = r'C:\\taylorkang\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'


# Open the player coordinates CSV file
playerData = open("./playercoordinates.csv")
playerDataReader = csv.reader(playerData)

# Read the header and determine the number of players
header = next(playerDataReader)
numPlayers = int((.5) * (len(header) - 1))

# Initialize global variables for player positions
xdatas = [0] * numPlayers
ydatas = [0] * numPlayers

# Create the frisbee field function
def generate_field():
    field = patches.Rectangle((0, 0), 120.0, 40.0, linewidth=2,
                              edgecolor='white', facecolor='green', zorder=0)

    fig, ax = plt.subplots(1, figsize=(11, 4))
    ax.add_patch(field)
    ax.axvline(x=25, color="white", zorder=1)
    ax.axvline(x=95, color="white", zorder=1)
    ax.axhline(y=0.0, color="white", zorder=1)
    ax.axhline(y=40.0, color="white", zorder=1)
    plt.axis('off')

    ax.scatter([], [], c='blue', label='Cutrules', zorder=2)
    ax.scatter([], [], c='red', label='Losing team', zorder=2)
    ax.legend(loc='upper right')

    return fig, ax

# Plot the static field
fig, ax = generate_field()
playerList = [ax.plot([], [], color=(1.0, 0.0, 0.0), marker='o')[0] for _ in range(numPlayers)]

# Function to update player positions in each frame
def update(frame):
    nextData = next(playerDataReader, None)

    if nextData:
        for i in range(1, len(header), 2):
            playerVal = int((.5) * (i - 1))
            xdatas[playerVal] = float(nextData[i]) / 10
            ydatas[playerVal] = float(nextData[i + 1]) / 10
            playerList[playerVal].set_data(xdatas[playerVal], ydatas[playerVal])

    return playerList

# Create the animation
# anim = animation.FuncAnimation(fig, update, frames=range(1, 100), repeat=False, interval=100)
anim = animation.FuncAnimation(fig, func=update, frames=range(1, 100), interval=100, repeat=True, save_count=1500)

# Add the ability to pause, play, and replay
def on_key(event):
    if event.key == 'p':
        anim.event_source.stop()
    elif event.key == 'r':
        anim.event_source.start()
    elif event.key == 's':
        anim.save('animation.mp4', writer='ffmpeg', fps=10)

fig.canvas.mpl_connect('key_press_event', on_key)

# Add a start/stop button
# ax_button = plt.axes([0.8, 0.01, 0.1, 0.05])
# button = Button(ax_button, 'Start/Stop', color='lightgoldenrodyellow', hovercolor='0.975')

# def on_button_click(event):
#     if anim.running:
#         anim.event_source.stop()
#     else:
#         anim.event_source.start()

# button.on_clicked(on_button_click)

f = r"c://taylorkang/Desktop/animation.mp4" 
writervideo = animation.FFMpegWriter(fps=60) 
anim.save(f, writer=writervideo)

# Show the animation
plt.show()