import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

recall = np.array([0.973708, 0.985301, 0.966121, 0.927344, 0.891355])

noCounts = np.array([
    [412, 431, 417, 452, 436],
    [412, 387, 363, 424, 425],
    [539, 516, 395, 500, 530],
    [383, 400, 282, 398, 426],
    [616, 604, 395, 511, 565]
])

noTrueCounts = np.round(noCounts / recall)

filepaths = [
    './results/D2.mat',
    './results/D3.mat',
    './results/D4.mat',
    './results/D5.mat',
    './results/D6.mat'
]

for i,filepath in enumerate(filepaths):
    # Load the predictions
    data = sio.loadmat(filepath)
    labels = data['Class']

    uniqueLabels, predictedCounts = np.unique(labels, return_counts=True)

    # plot a bar chart of the predicted counts and the actual counts
    fig, ax = plt.subplots()
    ax.bar(uniqueLabels, predictedCounts, label='Predicted')
    ax.bar(uniqueLabels, noTrueCounts[i], label='Actual')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title(f'D{i+2}')
    plt.show()
    