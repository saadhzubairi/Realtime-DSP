from matplotlib import pyplot

x = [1,2,3,4,5,6,7,8]
y = [0,3,3,5,4,6,4,1]
z = [0,1,6,7,8,9,4,12]

pyplot.figure(1)

graph_list  = pyplot.plot(x,y, x, z)

graph_list[0].set_color('#0096a4')
graph_list[0].set_linewidth('3')
graph_list[0].set_marker('o')
graph_list[0].set_linestyle('-')

graph_list[1].set_color('#9100a7')
graph_list[1].set_linewidth('2')
graph_list[1].set_marker('x')
graph_list[1].set_linestyle('--')

# can also do:
pyplot.setp(graph_list[0], color='blue', linestyle = '--')
pyplot.setp(graph_list[1], linewidth='3')

pyplot.xlim(0,10)
pyplot.ylim(0,12)

print(graph_list)

pyplot.show()
