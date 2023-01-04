import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

conf_matrix = np.array([[2792.0, 12.0, 12.0, 1.0, 4.0, 18.0, 73.0, 10.0, 17.0, 1.0], [1.0, 3328.0, 15.0, 3.0, 4.0, 3.0, 11.0, 4.0, 36.0, 0.0], [14.0, 20.0, 2825.0, 53.0, 5.0, 5.0, 10.0, 71.0, 91.0, 2.0], [6.0, 4.0, 26.0, 2816.0, 3.0, 83.0, 11.0, 22.0, 47.0, 12.0], [3.0, 17.0, 6.0, 3.0, 2751.0, 6.0, 63.0, 17.0, 18.0, 62.0], [22.0, 4.0, 6.0, 53.0, 1.0, 2497.0, 58.0, 12.0, 15.0, 8.0], [29.0, 16.0, 6.0, 0.0, 5.0, 11.0, 2792.0, 1.0, 7.0, 7.0], [3.0, 27.0, 48.0, 10.0, 8.0, 1.0, 4.0, 2908.0, 44.0, 31.0], [24.0, 34.0, 9.0, 33.0, 14.0, 50.0, 19.0, 21.0, 2709.0, 9.0], [15.0, 26.0, 7.0, 19.0, 47.0, 57.0, 9.0, 66.0, 69.0, 2712.0]])

# Convert to x_pred and y_pred 
y_pred = []
y_true = []
for x, one_arr in enumerate(conf_matrix): 
    for y, element in enumerate(one_arr): 
        for _ in range(int(element)):
            y_pred.append(y)
            y_true.append(x)

ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)

print(classification_report(y_true, y_pred, target_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))

plt.show()