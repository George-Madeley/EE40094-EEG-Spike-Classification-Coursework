import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

from preprocessing import loadPredictionData, createDataFrame, normalizeMax, highPassFilter, lowPassFilter

results_filepath = './results/D2.mat'
data_filepath = './data/D2.mat'

results = sio.loadmat(results_filepath, squeeze_me=True)
data = sio.loadmat(data_filepath, squeeze_me=True)

d = data['d']
labels = results['Class']
indices = results['Index']

low_cutoff_freq = 1000
high_cutoff_freq = 1
sampling_freq = 25000

df = createDataFrame(d)

# normalize the data
df_norm = normalizeMax(df)

# filter the data. Filtering can cause the amplitudes to decrease in
# power so the data is normalized again.
df_low_filtered = lowPassFilter(df_norm, low_cutoff_freq, sampling_freq)
# df_high_filtered = highPassFilter(df_low_filtered, high_cutoff_freq, sampling_freq)
df_filtered = normalizeMax(df_low_filtered)

start_idx = 40000
num_samples = 10000

# get the times and amplitudes for the first num_samples starting at start_idx
times = df_filtered['Time'][start_idx:num_samples + start_idx]
amplitudes = df_filtered['Amplitude'][start_idx:num_samples + start_idx]
plt.plot(times, amplitudes)

# get all the locations in indices where the value is less than num_samples + start_idx
# and greater than start_idx
locations = np.where((indices < num_samples + start_idx) & (indices >= start_idx))
capped_indices = indices[locations]
capped_labels = labels[locations]

# plot markers on the graph at the time times specified by the capped_indices.
# The markers should be different colors depending on the label. For example,
# if the label is 1, then the marker should be red. If the label is 2, then the
# marker should be blue.
for i in range(len(capped_indices)):
    # plot crosses
    plt.plot(df_filtered['Time'][capped_indices[i]], df_filtered['Amplitude'][capped_indices[i]], 'x', color='red' if capped_labels[i] == 1 else 'blue')

plt.show()
