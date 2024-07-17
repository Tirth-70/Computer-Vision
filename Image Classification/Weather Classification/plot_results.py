import pandas as pd
import matplotlib.pyplot as plt

# Load the results
results = pd.read_csv('./Image Classification/Weather Classification/yolo logs/train5/results.csv')
# print(results)
# Plot the results
plt.figure()
plt.plot(results['             train/loss'], label='train/loss')
plt.plot(results['               val/loss'], label='val/loss')
plt.grid()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(results['  metrics/accuracy_top1'], label='metrics/accuracy_top1')
plt.grid()
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
