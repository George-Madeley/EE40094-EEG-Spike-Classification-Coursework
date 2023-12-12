import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import loadTrainingData, createDataFrame, normalizeMax, lowPassFilter

window_size = 50

# Load the data
d, index, label = loadTrainingData()
df = createDataFrame(d, index, label)

# Normalize the amplitudes
df = normalizeMax(df)

df = lowPassFilter(df, 1000, 25000)

df = normalizeMax(df)

df_window = pd.concat([
    df['Amplitude'].shift(-i, fill_value=0) for i in range(-window_size, window_size)
], axis=1)
df_window.columns = [f'Amplitude{i}' for i in range(2 * window_size)]

# Remove the Ampitude column from the dataframe
df = df.drop(columns=['Amplitude'])

# Concatenate the dataframes
df = pd.concat([df, df_window], axis=1)

# remove record with label 0
df = df[df['Label'] != 0]

plt.subplots(2, 3)
# group the rows by the label column
grouped = df.groupby('Label')
# plot the first 20 windows for each label on the same graph
for i, (label, group) in enumerate(grouped):
    # get the values in the amplitude columns
    amplitudes = group.filter(regex='Amplitude\d+').values
    plt.subplot(2, 3, i + 1)
    # plot the amplitudes
    plt.plot(np.arange(2 * window_size), amplitudes[:100, :].T, label=f'Class {label}')
    plt.xlabel('Window Index')
    plt.ylabel('Amplitude')
    plt.title(f'Class {label}')

print("Waiting to show plot...")

plt.show()
