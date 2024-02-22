import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

fig, ax = plt.subplots()
xdata, ydata = 0.0, 0.0
ln, = ax.plot([], [], 'bo')

#set up csvList
playerData = open("./smoothData1.csv")
playerDataReader = csv.reader(playerData)
#clear header, store for debugging
header = next(playerDataReader)

def init():
    ax.set_xlim(0, 1100)
    ax.set_ylim(0,400)
    return ln,

def update(frame):
    nextData = next(playerDataReader)
    
    if (nextData) :
        xdata = (float(nextData[1]))
        ydata = (float(nextData[2]))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=range(1,100),
                    init_func=init, blit=True)
plt.show()