import matplotlib.pyplot as plt

x1_value = [1, 2, 3, 4, 5]
y1_value = [x ** 3 for x in x1_value]
x2_value = range(1, 5001)
y2_value = [x ** 3 for x in x2_value]

fig, ax = plt.subplots(1, 2)

ax[0].scatter(x1_value, y1_value, c=y1_value, cmap=plt.cm.Greens)
ax[1].scatter(x2_value, y2_value, c=y2_value, cmap=plt.cm.Reds)

ax[0].set_title("First Plot Title")  
ax[1].set_title("Second Plot Title") 
ax[0].set_xlabel('Value')
ax[1].set_xlabel('Value')
ax[0].set_ylabel('Cube of Value')
ax[1].set_ylabel('Cube of Value')

plt.show()
