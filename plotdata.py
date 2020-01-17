import matplotlib.pyplot as plt, numpy as np, sys, os
data = np.loadtxt(os.path.join(sys.path[0], "plotdata.txt"),delimiter=',',skiprows=1)

timedata = [elem[2] for elem in data]
xtick = ["{}x{}".format(int(elem[0]),int(elem[1])) for elem in data]
xtick.reverse()
plt.plot(timedata,xtick)
plt.savefig(os.path.join(sys.path[0], "graph.png"))
