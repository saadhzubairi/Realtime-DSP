# Update plot using interactive mode (ion)

import matplotlib
from matplotlib import pyplot
import math

matplotlib.use('TkAgg')
# matplotlib.use('MacOSX')		# Optional if using Mac OSX

print('The matplotlib backend is %s' % matplotlib.get_backend())  # plotting backend

my_fig = pyplot.figure(1)
my_ax = my_fig.add_subplot(1, 1, 1)   # axes

my_ax.set_xlim(0, 100)
my_ax.set_ylim(-1.2, 1.2)

pyplot.ion()  # Turn on interactive mode  

for x in range(100):
	my_ax.plot(x, math.sin(x/10), 'ro')  # plot each point as a red dot
	# pyplot.draw()        	 # Show result (not always needed)
	pyplot.pause(0.01)

pyplot.ioff()	# Turn off interactive mode
pyplot.show()
