#animation run 
#perhaps try saving stills instead of videos 
#deal with lost tracking

# help, instructions
help: 
	@echo "Welcome to Ultimetrics!"
	@echo "First, make sure to type 'make setup' in the command line to download all necessary software."
	@echo "To get started analyzing film, please add your video in the form of an MP4 file to this folder, and name it 'frisbee.mp4'."
	@echo "Then, you can type 'make run' into the command line'."

# gets everything necessary downloaded, assume pip is downloaded
# anything we need to install for animation? 
setup:
	@pip3 install opencv-python
	@pip3 install opencv-contrib-python

# run whole program
run: fieldtracking animation

# have user mark players
fieldtracking: 
	@echo "Please mark a box around each corner and each player"
	@echo "Note: this program works slowly, so outline a player, press enter, and then wait for the box to change color before marking the next player."
	@python3 tracking.py

# send to animation to be animated 
animation: 
	@echo "Animation phase."
	@python3 timescaleAnimate.py