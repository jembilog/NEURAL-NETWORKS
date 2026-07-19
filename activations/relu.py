import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

def relu(z):
    return np.maximum(np.zeros_like(z), z)
z = np.linspace(-8, 8 , 1000)
y = relu(z)
plt.plot(z, y)
plt.xlabel('z')
plt.ylabel('y(z)')
plt.grid()
plt.show()
