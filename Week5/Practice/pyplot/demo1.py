from matplotlib import pyplot

x = [1,2,3,4,5,6,7,8]
y = [0,3,3,5,4,6,4,1]
z = [0,1,6,7,8,9,4,12]

my_fig = pyplot.figure(1)

pyplot.plot(x,y,'ro--')
pyplot.plot(x,z,'gx--')
pyplot.xlabel('Time (n)')
pyplot.ylabel('sale (units/million)')

pyplot.show()

my_fig.savefig("pyplot_demo.pdf")