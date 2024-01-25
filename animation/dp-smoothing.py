import csv

#############
# This is the first iteration of a potential Dynamic Programming smoothing version. The first thing to do
# will be to, for all values that both have tracking and have their neighbors tracked, determine their 
# "slope" of sorts, which we can then use to extrapolate missing data
# The first, messy, way to do this might be using "prev line" and "2nd prev line" data, in some sort of 2D
# array, to begin creating these data values
##############

## ASSUMPTIONS
#file passed in is a csv with x and y coordinates of tracked players. One row for every frame, with a number
#of pairs of coordinates depending on the number of players. 


#set up csv reader 
def smoothVer1(fileName) :
    nextLineToWrite = ""
    with open(fileName) as playerCoords :
        
        outputName = fileName + "_output"
        with open(outputName, 'w') as outputFile:
            csvwriter = csv.writer(outputFile)
            #debugging
            #print(outputName)
            #test how many lines there are
            header = next(playerCoords)
            #re-write the header line, or at least skip it in the output file
            #debugging
            print(header)
            print(header.strip('\n'))
            print(header.strip('\n').split(','))
            csvwriter.writerow(header.strip('\n').split(','))
            csvwriter.writerow('testing')
            print(header.split(','))
            
            #debugging
            #print(len(header.split(',')))
            #this can accurately detect how many columns there are

            #initialize - it's not super efficient, but we'll see if it works
            twoBack = next(playerCoords).split(',')
            #maybe write this to the output file?
            oneBack = next(playerCoords).split(',')
            #maybe also write this to the output file?
            #some of the issue is that we need to read "into the future", makes DP difficult
            cur = next(playerCoords).split(',')
            oneForward = next(playerCoords).split(',')
            twoForward = next(playerCoords).split(',')
            #debugging
            #print(twoBack)
            #print(twoBack[2])
            #print(int(twoBack[2]))
            #print(oneBack)

            outputFile.close()
        playerCoords.close()
        


smoothVer1('smoothData1.csv')