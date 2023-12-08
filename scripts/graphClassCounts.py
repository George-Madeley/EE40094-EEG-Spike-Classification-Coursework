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


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 5))

for i,filepath in enumerate(filepaths):
    # Load the predictions
    data = sio.loadmat(filepath)
    labels = data['Class']

    uniqueLabels, predictedCounts = np.unique(labels, return_counts=True)

    predictionPercentages = predictedCounts / predictedCounts.sum() * 100
    actualPercentages = noTrueCounts[i] / noTrueCounts[i].sum() * 100

    # plot the percentages on a bar chart where the x-axis is the label and the
    # y-axis is the percentage. Each pair of bars should be adjacent to each
    # other. The left bar should be the actual percentage and the right bar
    # should be the predicted percentage.
    x = np.arange(len(uniqueLabels))
    width = 0.35

    ax[i // 3, i % 3].bar(x - width/2, actualPercentages, width, label='Actual')
    ax[i // 3, i % 3].bar(x + width/2, predictionPercentages, width, label='Predicted')

    ax[i // 3, i % 3].set_title(f'Class Counts for D{i+2}')
    ax[i // 3, i % 3].set_xticks(x)
    ax[i // 3, i % 3].set_xticklabels([f'Class {label}' for label in uniqueLabels])
    ax[i // 3, i % 3].legend()

plt.show()
    