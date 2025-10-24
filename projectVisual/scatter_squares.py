import matplotlib.pyplot as plt

plt.style.use('ggplot')

"""
x_values = [1, 2, 3, 4, 5]
y_values = [1, 4, 9, 16, 25]
"""

x_values = range(1, 1001)
y_values = [x ** 2 for x in x_values]


fig ,ax = plt.subplots()
# ax.scatter(x_values, y_values, s=10) 
ax.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, s=10)

# 设置图题和标签，但不指定 fontsize 和 labelsize
ax.set_title("Square Numbers")
ax.set_xlabel("Value")
ax.set_ylabel("Square of Value")

# 设置每个坐标轴的取值范围
# python会自动忽略_
ax.axis([0, 1100, 0, 1_100_000])

# ax.tick_params(labelsize=14)

plt.savefig('images/squares_plot.png', bbox_inches='tight')
plt.show()
