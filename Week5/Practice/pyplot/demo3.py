from matplotlib import pyplot
import math
pyplot.ion() #interactive mode on
pyplot.figure(1)

plots =  pyplot.plot([],[])

pyplot.xlim(0,200)
pyplot.ylim(-1.2,+1.2)

for x in range(1,200):
    pyplot.plot(x,math.sin(2*math.pi*x/25),'rx-')
    pyplot.pause(0.0001)

pyplot.show()
