import matplotlib.pyplot as plt
from random_walk import RandomWalk

rw = RandomWalk()
rw.fill_walk()

fig, ax = plt.subplots(figsize=(15, 9))
point_numbers = range(rw.num_points)

ax.plot(rw.x_values, rw.y_values, linewidth=1)

plt.show()
