import math

def compare() :
    #we assume that the two files passed in should have the same number of lines, if they don't then we can throw an exception
    trackerB = open("coordsListBoosting.txt", "r")
    trackerC = open("coordsListCSRT.txt", "r")
    trackerM = open("coordsListMIL.txt", "r")
    trackedCoordsListB = trackerB.readlines()
    trackedCoordsListC = trackerC.readlines()
    trackedCoordsListM = trackerM.readlines()

    truth = open("groundTruth.txt", "r")
    truthCoordsList = truth.readlines()
    #close and reopen the file to be at the top again
    truth.close()
    truth = open("groundTruth.txt", "r")

    #initialize statistics variables
    num = 0
    xTotalDiffB = 0.0
    xSquaredTotalDiffB = 0.0
    yTotalDiffB = 0.0
    ySquaredTotalDiffB = 0.0
    xTotalDiffC = 0.0
    xSquaredTotalDiffC = 0.0
    yTotalDiffC = 0.0
    ySquaredTotalDiffC = 0.0
    xTotalDiffM = 0.0
    xSquaredTotalDiffM = 0.0
    yTotalDiffM = 0.0
    ySquaredTotalDiffM = 0.0

    #now we know the lists are the same length
    for i in range(len(trackedCoordsListB)):
        truthXY = truthCoordsList[i].split(',')
        truthXY[1] = truthXY[1].strip('\n')
        trackedXYb = trackedCoordsListB[i].split(',')
        trackedXYb[1] = trackedXYb[1].strip('\n')
        trackedXYc = trackedCoordsListC[i].split(',')
        trackedXYc[1] = trackedXYc[1].strip('\n')
        trackedXYm = trackedCoordsListM[i].split(',')
        trackedXYm[1] = trackedXYm[1].strip('\n')
        
        #update stats
        num += 1
        xTotalDiffB += abs(float(trackedXYb[0]) - float(truthXY[0]))
        yTotalDiffB += abs(float(trackedXYb[1]) - float(truthXY[1]))
        xSquaredTotalDiffB += abs(float(trackedXYb[0]) - float(truthXY[0])) *  abs(float(trackedXYb[0]) - float(truthXY[0]))
        ySquaredTotalDiffB += abs(float(trackedXYb[1]) - float(truthXY[1])) * abs(float(trackedXYb[1]) - float(truthXY[1]))
        xTotalDiffC += abs(float(trackedXYc[0]) - float(truthXY[0]))
        yTotalDiffC += abs(float(trackedXYc[1]) - float(truthXY[1]))
        xSquaredTotalDiffC += abs(float(trackedXYc[0]) - float(truthXY[0])) *  abs(float(trackedXYc[0]) - float(truthXY[0]))
        ySquaredTotalDiffC += abs(float(trackedXYc[1]) - float(truthXY[1])) * abs(float(trackedXYc[1]) - float(truthXY[1]))
        xTotalDiffM += abs(float(trackedXYm[0]) - float(truthXY[0]))
        yTotalDiffM += abs(float(trackedXYm[1]) - float(truthXY[1]))
        xSquaredTotalDiffM += abs(float(trackedXYm[0]) - float(truthXY[0])) *  abs(float(trackedXYm[0]) - float(truthXY[0]))
        ySquaredTotalDiffM += abs(float(trackedXYm[1]) - float(truthXY[1])) * abs(float(trackedXYm[1]) - float(truthXY[1]))


    #output the stdErr of the data
    XstandardErrorB = math.sqrt(xSquaredTotalDiffB / num)
    YstandardErrorB = math.sqrt(ySquaredTotalDiffB / num)
    XstandardErrorC = math.sqrt(xSquaredTotalDiffC / num)
    YstandardErrorC = math.sqrt(ySquaredTotalDiffC / num)
    XstandardErrorM = math.sqrt(xSquaredTotalDiffM / num)
    YstandardErrorM = math.sqrt(ySquaredTotalDiffM / num)
    print("Standard Error Boosting + Truth X Coordinates: " + str(XstandardErrorB))
    print("Standard Error Boosting + Truth Y Coordinates: " + str(YstandardErrorB))
    print("Standard Error CSRT + Truth X Coordinates: " + str(XstandardErrorC))
    print("Standard Error CSRT + Truth Y Coordinates: " + str(YstandardErrorC))
    print("Standard Error MIL + Truth X Coordinates: " + str(XstandardErrorM))
    print("Standard Error MIL + Truth Y Coordinates: " + str(YstandardErrorM))

if __name__ == "__main__" :
    compare()
    