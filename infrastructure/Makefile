#put trackers on cones to get coordinates at every frame and put those into a txt file 
#run fourpointsmath.py for each frame and each player 
#want a fun loading message/graphic between phases
#animation run 
#instead put in the bottom center coordinate for each player (tracking by feet)
#deal with lost tracking

# help, instructions
# pip install waiting before running
help: 
	@echo "Welcome to Ultimetrics!"
	@echo "To get started analyzing film, please add your video in the form of an MP4 file to this folder, and name it 'frisbee.mp4'."
	@echo "Then, you can type 'make run' into the command line'."

run: coordinates markplayers math animation present

# have user enter coordinates of field
coordinates: 
	@echo "Please click the corners of the field in the following order: top left, top right, bottom right, bottom left."
	@python3 inputCoordinates.py
	@echo "The field coordinates are: "
	@cat corners.txt
	@echo

# have user mark players
markplayers: 
	@echo "Please mark a box around each player, as well as the disc."
	@echo "Note: this program works slowly, so outline a player, press enter, and then wait for the box to change color before marking the next player."
	@python3 tracking.py
	@cat playercoordinates.txt
	@echo

# send output csv to math to get converted 
math: 
	@echo "Math phase."

# send to animation to be animated 
animation: 
	@echo "Animation phase."

# present animation to user 
present: 
	@echo "Presentation phase."