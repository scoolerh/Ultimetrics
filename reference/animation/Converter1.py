import csv
#not sure if we need matplotlib
#from matplotlib import *
import pandas as pd
import numpy as np

#Takes in the direct output of the math function - X and Y pairs, flipped (since the read is vertical and the 
#display is horizontal, with all of the "null" values replaced with NAN, and then interpolated by the pandas
#program. Originally, we'll output two files, one interpolated linearly and one quadratically

def convert(fileName) :
    #open the requisite file and set up the reader
    oldData = open(fileName, 'r')
    oldDataReader = csv.reader(oldData)

    #open the target file and set up the reader
    newData = open("smoothed_data", 'w')
    newDataWriter = csv.writer(newData)

    #regardless of the length of the input file, we want to create a Pandas series for each of the inputs
    #these come in sets of 2, but they're columns -- if they're not, we can split them, maybe in an earlier program.
    #regardless, it shouldn't be too hard to code

    #initialize a header value
    header = next(oldDataReader)
    #debugging
    print(header)
    length = len(header)
    #initialize our lists
    columnList = []

    #set up the panda series - we'll have an array of Series objects, in order of columns
    for i in range(length) :
        newSeries = pd.Series()
        columnList.append(newSeries)
    #debugging
    print(columnList)


    #put things into the pandas series
    nextLine = next(oldDataReader)
    print(nextLine)
    print(len(nextLine))
    while nextLine :
        for i in range(len(nextLine)):
            columnList[i]._append(pd.Series(float(nextLine[i])))
            #debugging
            #print("added " + nextLine[i])
            #this works
            for i in range(len(columnList)) :
                print(columnList[i])
        nextLine = next(oldDataReader)
    #debugging
    print(columnList)

#testing
convert("./MockPlayerData.csv")