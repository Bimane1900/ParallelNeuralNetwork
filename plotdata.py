import matplotlib.pyplot as plt, numpy as np, sys, os
#data = np.loadtxt(os.path.join(sys.path[0], "plotdata.txt"),delimiter=',')


timedata = []
data = np.arange(9).reshape(3,3)
#xtick = ["{}x{}".format(elem[0],elem[1]) for elem in data]
# xtick = []
# for elem in data:
#     xtick.append("{}x{}".format(elem[0],elem[1]))
#     timedata.append(elem[2])
plt.xticks(data[0], xtick)
plt.plot(xtick,timedata)
print(xtick)
plt.savefig('plot.png')