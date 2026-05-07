# Exoplanet Detection using Transit Method

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load light curve data
df = pd.read_csv("light_curve.csv")

# Plot brightness
plt.figure(figsize=(12,5))
plt.plot(df['time'], df['brightness'])

plt.title("Light Curve")
plt.xlabel("Time")
plt.ylabel("Brightness")

plt.show()

# Invert brightness for dip detection
inverted_signal = -df['brightness']

# Detect transit dips
dips, properties = find_peaks(
    inverted_signal,
    height=-0.98,
    distance=30
)

# Print exoplanet count
print("Possible Exoplanets Detected:", len(dips))

# Plot dips
plt.figure(figsize=(12,5))

plt.plot(df['time'], df['brightness'])
plt.plot(
    df['time'][dips],
    df['brightness'][dips],
    "ro"
)

plt.title("Detected Transit Signals")
plt.xlabel("Time")
plt.ylabel("Brightness")

plt.show()
