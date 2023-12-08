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

    predictionPercentages = predictedCounts / predictedCounts.sum() * 100
    actualPercentages = noTrueCounts[i] / noTrueCounts[i].sum() * 100

    # plot the percentages on a bar chart where the x-axis is the label and the
    # y-axis is the percentage. Each pair of bars should be adjacent to each
    # other. The left bar should be the actual percentage and the right bar
    # should be the predicted percentage.
    x = np.arange(len(uniqueLabels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, actualPercentages, width, label='Actual')
    rects2 = ax.bar(x + width/2, predictionPercentages, width, label='Predicted')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Predicted vs Actual for D{i+2}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in uniqueLabels])
    ax.legend()
    plt.show()
    