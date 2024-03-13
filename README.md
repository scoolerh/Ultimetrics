# Ultimetrics

A Computer Science comps project by Ethan Ash, Conor Babcock O'Neill, Jack Huffman, Taylor Kang, Doug Pham, and Hannah Scooler. Advised by Professor Eric Alexander at Carleton College. The goal of this program is to track ultimate frisbee players from drone footage and convert it into a 2d animation.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following:

```bash
pip install yolov5
pip install opencv-contrib-python==4.9.0.80
pip install matplotlib
pip install ffmpeg
pip install PyQt5
pip install pandas
```

## Usage
Place an mp4 file name 'frisbee.mp4' in the project directory.
Run the following command while in the project directory:
```python
python3 ultimetrics.py
```
If you would like to view the tracking process while the program runs (which will take significantly longer to run), then set
```python
test_against_ground_truth = Truth
```
on line 639 of ultimetrics.py before running.

Full readme and directory can be found here https://docs.google.com/document/d/1zZKS5rFvqNjbPdI5Rey-tp2sGcvSVfJuEL2VXnOxmZY/edit?usp=sharing

## Testing
In order to get the accuracy of the program, set
```python
test_against_ground_truth = Truth
```
on line 641 of ultimetrics.py and then run
```python
python3 ground_truth.py
```
where you can annotate the ground truth location of all the players for every 20 frames. You can change the frequency of frames on line 48 in ground_truth.py and on line 642 in ultimetrics.py

Now, when you run 
```python
python3 ultimetrics.py
```
it will print out the average pixel distance away that the tracker is for every ground_truth frame in the video

## Contributing

For questions about the project or contributing to the project, reach out to [Eric Alexander](https://cs.carleton.edu/faculty/ealexander/).
