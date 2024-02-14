# smoothing player trajectories - jack 
# figure out how to avoid dependency problems - not priority (docker, server, documentation)

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
run: 
	@echo "ULTIMETRICS"
	@python3 tracking.py
	@python3 animation.py