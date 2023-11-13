import sys
import math

def compare() :
    #we assume that the two files passed in should have the same number of lines, if they don't then we can throw an exception
    tracker = open("coordsList.txt", "r")
    trackedCoordsList = tracker.readlines()

    #testing
    #print(trackedCoordsList)
    #print('ding')
    #for line in trackedCoordsList: 
        #print(line)
    truth = open("groundTruth.txt", "r")
    truthCoordsList = truth.readlines()
    #close and reopen the file to be at the top again
    truth.close()
    truth = open("groundTruth.txt", "r")

    #initialize statistics variables
    num = 0
    xTotalDiff = 0.0
    xSquaredTotalDiff = 0.0
    yTotalDiff = 0.0
    ySquaredTotalDiff = 0.0

    #test if the lists are the same length
    if len(trackedCoordsList) != len(truthCoordsList) :
        print("something went wrong")

    #now we know the lists are the same length
    else :
        for line in trackedCoordsList :
            #clean up data
            trackedXY = line.split(',')
            trackedXY[1] = trackedXY[1].strip('\n')
            
            #tester
            #print(trackedXY)
            #trackedXY is properly recognized

            truthXY = truth.readline().split(',')
            truthXY[1] = truthXY[1].strip('\n')
            
            #testing
            #print(truthXY)
            #truthXY is properly recognized
            
            #update stats
            num += 1
            xTotalDiff += abs(float(trackedXY[0]) - float(truthXY[0]))
            yTotalDiff += abs(float(trackedXY[1]) - float(truthXY[1]))
            xSquaredTotalDiff += abs(float(trackedXY[0]) - float(truthXY[0])) *  abs(float(trackedXY[0]) - float(truthXY[0]))
            ySquaredTotalDiff += abs(float(trackedXY[1]) - float(truthXY[1])) * abs(float(trackedXY[1]) - float(truthXY[1]))

            #testing
            
            # print(xTotalDiff)
            # print(yTotalDiff)
    #output the stdErr of the data
    XstandardError = math.sqrt(xSquaredTotalDiff / num)
    YstandardError = math.sqrt(ySquaredTotalDiff / num)
    print(XstandardError)
    print(YstandardError)


if __name__ == "__main__" :
    compare()
    