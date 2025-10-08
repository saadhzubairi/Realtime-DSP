# Example of pyplot

from matplotlib import pyplot

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]

# One plot
fig1 = pyplot.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)   # axes
ax1.plot(x, y)
ax1.set_xlabel('Time (n)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Data plot')

fig1.set_figheight(6.0)    # Comment/Uncomment and run again..

print('The figure height is', fig1.get_figheight())
print('The figure width is', fig1.get_figwidth())
  
pyplot.show()