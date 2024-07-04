# Image Classification for Parking Slots

This repository contains code for image classification of parking slots. The goal is to classify whether a parking slot is empty or occupied.

## Libraries

The following libraries are used in this project:

- `os`: for interacting with the operating system
- `pickle`: for serializing and deserializing Python objects
- `numpy`: for numerical computations
- `skimage.io`: for reading and writing images
- `skimage.transform`: for image resizing
- `sklearn.model_selection`: for model selection and evaluation
- `sklearn.svm`: for Support Vector Machine classification
- `sklearn.metrics`: for computing accuracy score

## Dataset

The dataset is located in the `/Image Classification/Parking slots/data` directory. It consists of two categories: `empty` and `occupied`. Each category contains images of parking slots.

## Data Preparation

The dataset is prepared by loading the images, resizing them to a fixed size of 15x15 pixels, and flattening them into a 1D array. The labels are assigned based on the category of the image.

## Model Training

The dataset is split into training and testing sets using a test size of 20%. A Support Vector Machine (SVM) classifier is trained using the training set. Grid search is performed to find the best combination of hyperparameters for the SVM classifier.

## Model Evaluation

The best model obtained from grid search is evaluated on the testing set. The accuracy of the model is computed using the `accuracy_score` function from the `sklearn.metrics` module. The accuracy of the model is 97.1%.

## Model Saving

The best model is saved as a pickle file named `model.p` using the `pickle.dump` function.

For more details, refer to the code in the repository.

