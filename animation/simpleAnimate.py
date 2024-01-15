#import the requisite modules
#for this first simple trial animation, we will try using Matplotlib. There's not a ton out there about
#how to animate, so the next thing to look into might be timescaleDB. That one seems a bit more convoluted
#another promising avenue is the graphics library
#you need to install matplotlib and graphics

from graphics import *
import csv
import time

#field can be drawn using GraphWin
#we want our dimensions to be quite large -
#we can go with 1100 by 400, 10 times the dimensions of a frisbee field

#output: a field object that can then be modified
def drawField() :
    field = GraphWin("Field", 1100, 400)
    field.setBackground("lime")
    
    #draw the line on the field
    leftEndzoneLine = Line(Point(200,0),Point(200,400))
    leftEndzoneLine.draw(field)
    rightEndzoneLine = Line(Point(900,0),Point(900,400))
    rightEndzoneLine.draw(field)
    #debugging/testing
    #testCirc = Circle(Point(300.0, 400.0), 100.0)
    #testCirc.draw(field)
    #for testing and making sure the field doesn't close
    #debugging
    field.getMouse()
    #field.close()
    return field

#input: a field object, and a csv file of player locations
#output: an animated field, displayed on the screen
#Conor's notes: I think for this time, we will try using circles, since that will be simplest. 
#This program is mostly meant to test what we want it to look like, and try and translate that 
#into matplotlib, to be able to save the file somewhere. 

#okay, now how do we do this for multiple players? 
#can we use a list of players? going to be tricky, maybe we can mush the players all into one giant csv file?
#is that heinous?
#the way we would do that is this (with 14 players worth of data, we can worry about the disc later)
#so assume 29 lines - the number of the frame, and then 14 x and 14 y data
    # for x in range (1, 30, 2) :
    #     player = Circle(Point(nextSpot[x], nextSpot[x + 1]), 20)
    #     player.setFill('White')
    #     player.draw(field)
def animate(field, playerDataList) :
    #playerDataList should be a CSV file 
    locationData = open(playerDataList)
    locationDataReader = csv.reader(locationData)
    #remove header, store for debugging
    header = next(locationDataReader)
    #debugging
    print(header)

    #create the while loop that will give us position
    #for now we just want one circle, we can make this modular and have more players later
    nextSpot = next(locationDataReader)
    #start with a null player
    player = Circle(Point(-1, -1), 0)
    while (nextSpot) :
        player.undraw()
        player = Circle(Point(nextSpot[1], nextSpot[2]), 20)
        player.setFill('White')
        player.draw(field)
        #pause for animation
        #this can be modular for how smooth we want the animation to be 
        #maybe we don't want any waiting at 30 fps
        time.sleep(.05)
        #debugging
        # field.getMouse()
        #this isn't perfect, because of the flickering
        #and it is NOT fast enough
        nextSpot = next(locationDataReader)
        #debugging
        print(nextSpot)
    field.close()
    return

test = drawField()
animate(test, "/Users/cslab/Desktop/Ultimetrics/animation/smoothData1.csv")
