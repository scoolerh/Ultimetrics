import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import csv

##use the savgol filter to smooth large amounts of data
#look at time spent, how well the smoothing works
#we're going to smooth each column individually
playerDataReader = csv.reader(open("../playercoordinates.csv"))
numColumns = len(next(playerDataReader, False))
#numColumns works correctly
#create our 2D array
data = [[] for x in range(numColumns)]

#initialize a global variable
numRows = 0
for row in playerDataReader :
    #print(row)
    #increment
    
    numRows += 1
    for col in range(numColumns) :
        #print(col)
        data[col].append(float(row[col]))
#data is correctly initialized

#create an array of numpy arrays
numpyData = []
#create an array of smoothed data
smoothedData = []
for col in range(numColumns) :
    
    numpyData.append(np.array(data[col]))
    #these are our parameters to change
    smoothedData.append(savgol_filter(numpyData[col], 10, 3))
#debug
# print(data)
# print(smoothedData)


#the data at this stage is a list of numpy arrays, which we will need to read index by index until we've backfilled
#for now we just want to print line by line to see if we'd be adding the correct things
#initialize the next row
outfile = open("smoothedplayercoordinates.csv", 'w')
for row in range(numRows) :
    nextRow = []
    for col in smoothedData :
        #this is cursed but it might work
        nextRow.append(col[row])
    #print(nextRow)
    outfile.write(str(nextRow).strip('[]'))
    outfile.write('\n')
outfile.close()
        
