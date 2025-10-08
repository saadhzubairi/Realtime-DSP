# Set data after initial plot command

from matplotlib import pyplot

my_fig = pyplot.figure(1)
my_ax = my_fig.add_subplot(1, 1, 1)   # axes

[g1] = my_ax.plot([], [])   # Create empty graph

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]

my_ax.set_xlim(0, 10)
my_ax.set_ylim(0, 10)

g1.set_xdata( x )
g1.set_ydata( y )

my_ax.set_xlabel('x axis')
my_ax.set_ylabel('y axis')
my_ax.set_title('title')

pyplot.show()

