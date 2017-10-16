import matplotlib.pyplot as plt
import numpy as np


def PlotRMSEperMin(tester, algorithms=None, axes=None):
    prediction_window = tester.prediction_window
    if algorithms is None:
        algorithms = tester.TestedAlgorithms
    if axes is None:
        fig, axes = plt.subplots(figsize=(20, 15), dpi=400)

    rmse = tester.rmse()

    for a in algorithms:
        label = a + " Prediction Error"
        axes.plot(range(1, prediction_window), rmse[a][1:], label=label, linewidth=1.5)

    axes.legend(loc='lower right')


def PlotAverageRMSE(tester, algorithms=None, axes=None):
    if algorithms is None:
        algorithms = tester.TestedAlgorithms
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 5), dpi=400)

    algorithms = list(algorithms)

    armse = tester.average_rmse()

    results = [armse[key] for key in algorithms]

    axes.set_ylim(np.min(results) * 0.995, np.max(results) * 1.005)

    x_pos = np.arange(len(results))

    # for a in algorithms:
    # label = a + " Prediction Error"
    axes.bar(x_pos, results, align='center', alpha=0.4)
    axes.set_xticks(x_pos)
    axes.set_xticklabels(algorithms)


def PlotAverageTotalError(tester, algorithms=None, axes=None):
    if algorithms is None:
        algorithms = tester.TestedAlgorithms
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 5), dpi=400)

    algorithms = list(algorithms)

    ate = tester.average_total_error()

    results = [ate[key] for key in algorithms]

    axes.set_ylim(np.min(results) * 0.995, np.max(results) * 1.005)

    x_pos = np.arange(len(results))

    # for a in algorithms:
    # label = a + " Prediction Error"
    axes.bar(x_pos, results, align='center', alpha=0.4)
    axes.set_xticks(x_pos)
    axes.set_xticklabels(algorithms)


def PlotSimplePrediction(tester, offset, algorithms=None, axes=None):
    prediction_window = tester.prediction_window
    prediction = tester.simple_prediction(offset)

    if algorithms is None:
        algorithms = tester.TestedAlgorithms
    if axes is None:
        fig, axes = plt.subplots(figsize=(20, 10), dpi=400)

    for a in algorithms:
        label = a + " Prediction"
        axes.plot(range(prediction_window), prediction[a], label=label, linewidth=2)
        # print("RMSE of this ", a, " prediction is: ", mean_squared_error(prediction['GroundTruth'], prediction[a])**0.5)

    axes.plot(range(prediction_window), prediction['GroundTruth'], label='Ground Truth', linewidth=2)

    axes.legend()
