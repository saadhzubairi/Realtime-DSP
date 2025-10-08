# Define labels and create legend after initial plot command

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]
z = [5, 1, 4, 2, 2, 4]

my_fig = pyplot.figure(1)
my_ax = my_fig.add_subplot(1, 1, 1)   # axes

[g1] = my_ax.plot(x, y)
[g2] = my_ax.plot(x, z)

g1.set_label('apple')	# Set parameters of line1
g2.set_label('pear')	# Set parameters of line2
my_ax.legend()			# Create legend

pyplot.show()

