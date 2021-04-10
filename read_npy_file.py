# load numpy array from npy file
from numpy import load
import numpy as np
# load array



data = load('black.npy')
print("Data finished loading...")
print()
print("Looking for min and max values in the array")
#print the array
print(data[3, 6, 6, 0, 0, 0, 1, 0, 0, 1, 1])

print("Max value is:", np.max(data))
print("Min value is:", np.min(data))

print(np.unravel_index(np.argmax(data, axis=None), data.shape))