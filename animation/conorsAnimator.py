import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import csv


#assuming 30fps, if we want a minute of video, we want to animate for 30*60 = 1800 frames
#input is a data csv file for a given player, we can modify animate later to have it loop for 
#a number of players
#the csv file will have the frame number, x coord in terms of yards, and y coord in terms of yards in that order
def singlePLayerAnimate(data) :
    n = 1800
    locationData = open(data)
    locationDataReader = csv.reader(locationData)
    #skip the header, store it in case for debugging
    header = next(locationDataReader)

    #we are ready to animate

    #define metadata for movie
    # Define the meta data for the movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='animation test', artist='Conor',
                    comment='one player running on a field')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    #initialize the field 
    field = plt.figure(figsize=(11,4))

    #we will start with players as points, for now
    # Update the frames for the movie
    #initialize
    x = 0.0
    y = 0.0
    with writer.saving(fig, "writer_test.mp4", 100):
        for i in range(n):
            #grab the next line
            nextData = next(locationDataReader)
            if nextData  :
                x = (nextData[1] / 10)
                y = (nextData[2] / 10)
                plt.plot(x,y)
            writer.grab_frame()



