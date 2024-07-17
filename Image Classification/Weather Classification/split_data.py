import os
import splitfolders
file_path = 'Multi-class Weather Dataset'

print(os.listdir(file_path))

splitfolders.ratio(file_path,seed=1234, output="Image Classification/Weather Classification/data", ratio=(0.8, 0.2, 0))