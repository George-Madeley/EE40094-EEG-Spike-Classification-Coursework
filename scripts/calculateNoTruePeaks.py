import numpy as np

num_guesses = np.array([
    [36905, 71408, 234873, 405856, 501859],
    [1608, 1774, 5595, 3364, 9385],
    [3870, 16182, 19734, 20683, 20748]
])
recall = np.array([
    [0.973708, 0.985301, 0.966121, 0.927344, 0.891355],
    [0.728921, 0.813327, 0.810784, 0.4811, 0.462736],
    [0.944696, 0.956884, 0.955218, 0.964654, 0.943359]
])
precision = np.array([
    [0.058203, 0.028162, 0.010563, 0.004654, 0.005362],
    [1.0, 0.935738, 0.372118, 0.29132, 0.148855],
    [0.538501, 0.12069, 0.124303, 0.095006, 0.137266]
])

# calculate True Positives
TP = precision * num_guesses

# calculate False Negatives
FN = (TP / recall) - TP

# calculate the number of positives
P = np.ceil(TP + FN)

print(f'Number of Positives:\n{P}')