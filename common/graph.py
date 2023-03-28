import matplotlib.pyplot as plt
location = ['Restaurant','Laptop','Twitter']
x = range(len(location))
x2_width = ['Restaurant1','Laptop1','Twitter1']

plt.bar([i-0.1 for i in x], [727,337,336], lw=0.5, fc="b", width=0.2, label="Label")
plt.bar([i+0.1 for i in x], [699,300,283], lw=0.5, fc="y", width=0.2, label="predict")
plt.legend(['Label', 'predict'], fontsize=10, markerscale=0.5)
plt.xticks(x, location, fontsize=8)
plt.ylabel("Number", fontsize=8, rotation=90)
plt.xlabel("Negative", fontsize=8, rotation=0)
# plt.xticks(range(0, 5), x_data)
plt.show()