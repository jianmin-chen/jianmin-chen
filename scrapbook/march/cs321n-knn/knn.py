import cifar10
import matplotlib.pyplot as plt
import numpy as np

if not cifar10.get_path().is_file():
    cifar10.download()
else:
    print("cifar10 is already downloaded at:\n{}".format(cifar10.get_path()))

x_train, y_train, x_test, y_test = (i.astype("float32") for i in cifar10.load())
x_train = x_train.transpose([0, 2, 3, 1])
x_test = x_test.transpose([0, 2, 3, 1])

print("Training data shape: ", x_train.shape)
print("Training labels shape: ", y_train.shape)
print("Test data shape: ", x_test.shape)
print("Test labels shape: ", y_test.shape)
print("\n")

# Visualize some examples from the dataset
# We show a few examples of training images from each class.
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)
samples_per_class = 7

# 50,000 too much, use 5000 training and 500 test
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
print("Training data shape: ", x_train.shape)
print("Training labels shape: ", y_train.shape)
print("Test data shape: ", x_test.shape)
print("Test labels shape: ", y_test.shape)

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
print("new train-shape:", x_train.shape)
print("new test-shape:", x_test.shape)

def compute_distances(x, y):
    x_reshaped = x.reshape(x.shape[0], 1, x.shape[1])
    y_reshaped = y.reshape(1, y.shape[0], y.shape[1])
    inside = x_reshaped - y_reshaped
    print("Subtract")
    inside = inside ** 2
    print("Squared")
    inside = np.sum(inside, axis=2)
    print("Sum")
    return np.sqrt(inside)

dists = compute_distances(x_test, x_train)
