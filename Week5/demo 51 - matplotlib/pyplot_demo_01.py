# Example of pyplot

from matplotlib import pyplot

print('The matplotlib backend is %s' % pyplot.get_backend())  # The backend used plotting

x = [1, 3, 9, 8, 4, 6]
y = [6, 3, 6, 8, 5, 9]

# One plot
fig1 = pyplot.figure()
ax1 = fig1.add_subplot(1, 1, 1)   # axes
ax1.plot(x, y)
ax1.set_xlabel('Time (n)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Data plot')
  
pyplot.show()


[fig2, ax2] = pyplot.subplots()  # short version
ax2.plot(x, y, color = 'red')
ax2.set_xlabel('Time (n)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Data plot')

pyplot.show()

