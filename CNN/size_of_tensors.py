# %matplotlib inline	
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from mnist import MNIST
data = MNIST(data_dir="Data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))
print("- Test-set:\t\t{}".format(data.num_test))
data.y_test[0:5,:]