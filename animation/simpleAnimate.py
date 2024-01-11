#import the requisite modules
#for this first simple trial animation, we will try using Matplotlib. There's not a ton out there about
#how to animate, so the next thing to look into might be timescaleDB. That one seems a bit more convoluted
#another promising avenue is the graphics library
#you need to install matplotlib and graphics

from graphics import *

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
    #for testing and making sure the field doesn't close
    #debugging
    field.getMouse()
    field.close()
    return field

def animate(field, playerDataList) :
    
    return

test = drawField()
animate(test, "")
