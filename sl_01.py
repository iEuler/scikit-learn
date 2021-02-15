from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# iris = datasets.load_iris()
# digits = datasets.load_digits()

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %f' % label)
# plt.show()

X, y = datasets.load_iris(return_X_y=True)

clf = SVC(kernel='linear')
# clf.set_param(kernel='linear').fit(X[:-5], y[:-5])
clf.fit(X[:-5], y[:-5])
print(clf.predict(X[-5:]))
clf = SVC(kernel='rbf')
clf.fit(X[:-5], y[:-5])
print(clf.predict(X[-5:]))