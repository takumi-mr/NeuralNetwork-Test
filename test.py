import numpy as np
import matplotlib.pyplot as plt
import activateFunction as f

x = np.arange(-6, 6, 0.1)
y = f.relu(x)

plt.plot(x, y)
plt.show()
