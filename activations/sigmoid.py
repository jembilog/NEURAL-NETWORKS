import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
def sigmoid(z):
    return 1.0/  (1 + np.exp(-z))
z = np.linspace(-8, 8, 1000)
y = sigmoid(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.grid()
plt.show()
