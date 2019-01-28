
from matplotlib import pyplot
import pylab
#extract 2 arrays from delimiter spaced data
traindata = pylab.loadtxt('trainingdata.txt', delimiter=',')
x = traindata[:,0]
y = traindata[:,1]
#using scatter plot
pyplot.scatter(x,y)
pyplot.xlabel('Charged time in Hrs')
pyplot.ylabel('Laptop usage in Hrs')
pyplot.grid()
#display plot
pyplot.show()

charge_time = float(input().strip())
# charge_time should be positive
print("{0:.2f}".format(min(charge_time*2,8)))
