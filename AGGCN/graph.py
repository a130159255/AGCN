import matplotlib.pyplot as plt
import numpy

x_axis_data = [1,2,3,4,5,6,7,8,9,10]
y1 = [86.05,87.4,84.89,86.59,85.16,85.15,85.87,84.54,81.77,86.14]
y2 = [79.53,80.98,78.44,80.31,78.28,78.28,77.66,78.75,79.38,78.28]
y3 = [74.97,77.16,75.12,75.29,74.53,74.53,75.84,74.39,75.84,75.26]

# y1 = [79.29,81.39,78.91,81.24,78.65,78.18,79.57,77.81,71.64,79.98]
# y2 = [75.61,77.61,75.42,77.17,73.41,75.04,73.65,75.37,76.14,75.07]
# y3 = [74.12,74,73.69,73.8,73.71,73.38,73.9,72.82,74.51,73.75]

plt.plot(x_axis_data,y1,'b*--',alpha=0.5,linewidth=1,label='acc')
plt.plot(x_axis_data,y2,'rs--',alpha=0.5,linewidth=1,label='acc')
plt.plot(x_axis_data,y3,'go--',alpha=0.5,linewidth=1,label='acc')

plt.legend()
plt.xlabel("GCN layers")
plt.ylabel("Acc")
plt.show