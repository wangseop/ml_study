import numpy as np
import matplotlib.pyplot as plt

x = np.array(range(0, 10, 0.1))

y = np.sin(x)

plt.plot(x, y)

plt.show()