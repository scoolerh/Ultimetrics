# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import csv

# Read player coordinates from CSV
player_data = pd.read_csv("./playercoordinates.csv")

# Function to generate the frisbee field
def generate_field(ax):
    # Create the field
    field = patches.Rectangle((0, 0), 120.0, 40.0, linewidth=2,
                              edgecolor='white', facecolor='green', zorder=0)
    ax.add_patch(field)
    ax.axvline(x=25, color="white", zorder=1)
    ax.axvline(x=95, color="white", zorder=1)
    ax.axhline(y=0.0, color="white", zorder=1)
    ax.axhline(y=40.0, color="white", zorder=1)
    ax.axis('off')
    ax.scatter([], [], c='blue', label='Cutrules', zorder=2)
    ax.scatter([], [], c='red', label='Losing team', zorder=2)
    ax.legend(loc='upper right')

# Function to update plot for each frame
def update_plot(frame):
    ax.clear()
    generate_field(ax)
    # Convert frame to integer if necessary
    frame = int(frame)
    # Plot player positions
    for i in range(1, len(player_data.columns), 2):
        player_x = player_data.iloc[frame, i + 1]
        player_y = player_data.iloc[frame, i]
        ax.plot(player_x, player_y, 'ro')  # Assuming player positions are in columns 1,3,5,...
    return mplfig_to_npimage(fig)

# Define duration of the video
duration = len(player_data)

# Create figure and axis objects
fig, ax = plt.subplots(1, figsize=(11, 4))

# Generate the initial field
generate_field(ax)

# Create the animation
animation = VideoClip(update_plot, duration=duration)

# Display the animation with auto play and looping
animation.ipython_display(fps=20, loop=True, autoplay=True)