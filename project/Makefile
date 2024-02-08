# match ids of redetected players - ethan and conor
# smoothing player trajectories - jack 
# call HITL if not 14 players detected at the beginning of the video - hannah (label a thing on video with a single click, very reusable code)
# color average does not help, they are all very close
# figure out how to avoid dependency problems - not priority (docker, server, documentation)
# change color of boxes to enhance contrast, add numbers to each box - hannah and doug
# test yolov5 model - doug
# test on different videos, hard code cones for debugging 

# help, instructions
help: 
	@echo "Welcome to Ultimetrics!"
	@echo "First, make sure to type 'make setup' in the command line to download all necessary software."
	@echo "To get started analyzing film, please add your video in the form of an MP4 file to this folder, and name it 'frisbee.mp4'."
	@echo "Then, you can type 'make run' into the command line'."

# gets everything necessary downloaded, assume pip is downloaded
setup:
	@pip3 install yolov5
	@pip3 uninstall opencv-contrib-python
	@pip3 uninstall opencv-python
	@pip3 install yolov5

# run whole program 
run: fieldtracking animation

# have user mark players
fieldtracking: 
	@echo "Please mark a box around each corner."
	@python3 tracking.py

# send to animation to be animated 
animation: 
	@python3 animation.py