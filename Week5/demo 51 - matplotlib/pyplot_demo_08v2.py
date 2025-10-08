# Set data after initial plot command

from matplotlib import pyplot

my_fig = pyplot.figure(1)
my_ax = my_fig.add_subplot(1, 1, 1)   # axes

[g1] = my_ax.plot([], [], 'blue')  # Create empty graph
[g2] = my_ax.plot([], [], 'red')   # Create empty graph

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]
z = [5, 1, 4, 2, 2, 4]

g1.set_xdata( x )
g1.set_ydata( y )

g2.set_xdata( x )
g2.set_ydata( z )

my_ax.set_xlim(0, 10)
my_ax.set_ylim(0, 10)

pyplot.show()

