 # Specify line style, line width, markers
 # Specify axis limits

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]
z = [5, 1, 4, 2, 2, 4]

# Two different line styles 
[fig1, ax1] = pyplot.subplots()
ax1.plot(x, y, 'r-')			 # red solid line
ax1.plot(x, z, 'b--')		     # blue dashed line
ax1.set_xlabel('Time (n)')

pyplot.show()
# exit()

# Two different line styles with markers
[fig2, ax2] = pyplot.subplots()
ax2.plot(x, y, 'r-', marker = 'o')	# circle markers
ax2.plot(x, z, 'b--', marker = 's')	# square markers
ax2.set_xlabel('Time (n)')

pyplot.show()

# Two different marker styles 
[fig3, ax3] = pyplot.subplots(num = 8)
ax3.plot(x, y, 'ro-')		# red solid line with circle marker
ax3.plot(x, z, 'bs--')		# blue dashed line with square markers
ax3.set_xlabel('Time (n)')

pyplot.show()

# Specify marker size, line width
[fig4, ax4] = pyplot.subplots()
ax4.plot(x, y, 'r-', linewidth = 4)
ax4.plot(x, z, 'bo--', markersize = 18, linewidth = 3)
ax4.set_xlabel('Time (n)')

pyplot.show()

[fig5, ax5] = pyplot.subplots()
ax5.plot(x, y, 'r-')
ax5.plot(x, z, 'b--')
ax5.set_xlabel('Time (n)')
ax5.set_xlim(-2, 12)
ax5.set_ylim(0, 10)

pyplot.show()
exit()

