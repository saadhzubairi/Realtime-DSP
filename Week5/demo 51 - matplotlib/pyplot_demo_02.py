 # Two plots in one axis

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]
z = [5, 1, 4, 2, 2, 4]

# Specify colors
fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)   # axes
ax1.plot(x, y, color = 'red')
ax1.plot(x, z, color = 'blue')
ax1.set_xlabel('Time (n)')

# pyplot.show()

# Specify colors, short method
fig2 = pyplot.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)   # axes
ax2.plot(x, y, 'red')
ax2.plot(x, z, 'blue')
ax2.set_xlabel('Time (n)')

# pyplot.show()

# Create legend
fig3 = pyplot.figure(3)
ax3 = fig3.add_subplot(1, 1, 1)   # axes
ax3.plot(x, y, 'red', label = 'apples')
ax3.plot(x, z, 'blue', label = 'bananas')
ax3.set_xlabel('Time (n)')
ax3.legend()

pyplot.show()

