# Meteor Detection 

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


df = pd.read_csv("meteor_data.csv")


print(df.head())

# Plot signal
plt.figure(figsize=(12,5))
plt.plot(df['signal'], label='Signal')

# Detect peaks
peaks, properties = find_peaks(
    df['signal'],
    height=50,
    distance=20
)

# Print total number of meteors
print("Total Meteors Detected:", len(peaks))

# Plot detected peaks
plt.plot(peaks, df['signal'][peaks], "ro", label='Meteor Peaks')

# Labels
plt.title("Meteor Detection")
plt.xlabel("Time")
plt.ylabel("Signal Strength")
plt.legend()

# Show graph
plt.show()
