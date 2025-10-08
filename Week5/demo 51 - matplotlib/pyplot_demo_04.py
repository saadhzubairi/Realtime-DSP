# Save figure as pdf file

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]
z = [5, 1, 4, 2, 2, 4]

my_fig = pyplot.figure()
my_ax = my_fig.add_subplot(1, 1, 1)   # axes
my_ax.plot( x, y, 'r--', label = 'apples', linewidth = 2 )
my_ax.plot( x, z, 'y', label = 'bananas', linewidth = 3 )
my_ax.set_xlabel('Time (n)')
my_ax.legend()
pyplot.show()

my_fig.savefig('pyplot_demo_04_figure.pdf')
