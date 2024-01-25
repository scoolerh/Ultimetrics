from graphics import *

def testColor() :
    tester = GraphWin("test", 500, 500)
    tester.setBackground(color_rgb(255,100,100))
    
    # pt = Point(100,50)
    # pt.draw(tester)
    tester.getMouse()
    tester.setBackground(color_rgb(int(255 * .5),0, 0))
    tester.getMouse()
    
    tester.close()
    
    return tester

testColor()


#KEY TAKEAWAY FOR COLOR METHOD -
# For confidence value, we 