import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import csv

def smooth_position_data(position_data, window_size=10, order=2):
    smoothed_data = savgol_filter(position_data, window_size, order)
    return smoothed_data
def smooth_2(position_data, window_size=5, order=3):
    smoothed_data = savgol_filter(position_data, window_size, order)
    return smoothed_data
def smooth_3(position_data, window_size=5, order=2):
    smoothed_data = savgol_filter(position_data, window_size, order)
    return smoothed_data
# Example usage with a single column of position data
playerDataReader = csv.reader(open("OneColPlayerCoords.csv"))
arr = []

for row in playerDataReader:
    arr.append(float(row[0]))  # Convert the string to float

position_data = np.array(arr)  # Convert to NumPy array

smoothed_data = smooth_position_data(position_data)
smoothed_data2 = smooth_2(position_data)
smoothed_data3 = smooth_3(position_data)

# Plotting the original and smoothed data
plt.plot(position_data, label='Original Data')
plt.plot(smoothed_data, label='Smoothed Data (window 10, order 2)')
plt.plot(smoothed_data2, label="Smoothed Data (window 5, order 3)")
plt.plot(smoothed_data3, label="Smoothed Data (window 5, order 2)")

plt.legend()
plt.show()
