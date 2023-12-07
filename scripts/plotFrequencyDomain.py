from preprocessing import loadPredictionData, createDataFrame, plotFrequencyDomain, lowPassFilter, highPassFilter, normalizeAmplitudes, plotTimeDomain

import matplotlib.pyplot as plt

high_pass_cutoff_freq = 500
low_pass_cutoff_freq = 750


filepaths = [
    'data/D1.mat',
    'data/D2.mat',
    'data/D3.mat',
    'data/D4.mat',
    'data/D5.mat',
    'data/D6.mat',
]

# Create a subplot with 2 rows and 3 columns
fig, ax = plt.subplots(2, 3)

num_samples_plot = -1

for i, filepath in enumerate(filepaths):
    # Load the data
    d = loadPredictionData(filepath)

    # Create the dataframe
    df = createDataFrame(d)

    plt.subplot(2, 3, i + 1)

    # Plot the unfiltered data in the frequency domain
    plotFrequencyDomain(df, num_samples_plot, label='Unfiltered')

    # Apply a low pass filter to the data
    df = lowPassFilter(df, low_pass_cutoff_freq, sampling_freq=25000)

    # Plot the low pass filtered data in the frequency domain
    plotFrequencyDomain(df, num_samples_plot, label='Low Pass Filtered')

    # Apply a high pass filter to the data
    df = highPassFilter(df, high_pass_cutoff_freq, sampling_freq=25000)

    # Plot the high pass filtered data in the frequency domain
    plotFrequencyDomain(df, num_samples_plot, label='High Pass Filtered')

    plt.title(f'D{i + 1}.mat')

plt.show()

# Create a subplot with 2 rows and 3 columns
fig, ax = plt.subplots(2, 3)

num_samples_plot = 5000

for i, filepath in enumerate(filepaths):
    # Load the data
    d = loadPredictionData(filepath)

    # Create the dataframe
    df = createDataFrame(d)

    # Normalize the amplitudes
    df = normalizeAmplitudes(df)

    plt.subplot(2, 3, i + 1)

    # Plot the unfiltered data in the frequency domain
    plotTimeDomain(df, num_samples_plot, label='Unfiltered')

    # Apply a low pass filter to the data
    df = lowPassFilter(df, low_pass_cutoff_freq, sampling_freq=25000)

    # Normalize the amplitudes
    df = normalizeAmplitudes(df)

    # Plot the low pass filtered data in the frequency domain
    plotTimeDomain(df, num_samples_plot, label='Low Pass Filtered')

    # Apply a high pass filter to the data
    df = highPassFilter(df, high_pass_cutoff_freq, sampling_freq=25000)

    # Normalize the amplitudes
    df = normalizeAmplitudes(df)

    # Plot the high pass filtered data in the frequency domain
    plotTimeDomain(df, num_samples_plot, label='High Pass Filtered')

    plt.title(f'D{i + 1}.mat')

plt.show()
