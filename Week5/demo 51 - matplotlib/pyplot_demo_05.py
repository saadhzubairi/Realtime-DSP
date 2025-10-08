# Set parameters after initial plot command

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]

my_fig = pyplot.figure(1)
my_ax = my_fig.add_subplot(1, 1, 1)   # axes

[g1] = my_ax.plot(x, y)   # one graph

g1.set_color('red')
g1.set_linewidth('1')
g1.set_linestyle('--')
g1.set_marker('o')
my_ax.set_title('My Data')

pyplot.show()

