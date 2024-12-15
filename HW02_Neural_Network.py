import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 1. ใช้ make_blobs สร้างข้อมูล 2 Class
X, y = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], cluster_std=0.75, random_state=42)

# แบ่งข้อมูลเป็น Training และ Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. สร้าง Neural Network เพื่อ Classification
nn_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# 3. สร้าง Decision Boundary
def plot_decision_boundary(X, y, model):
    h = 0.02  # ความละเอียดของตารางตาข่าย
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.coolwarm)
    plt.title("Decision Plane")
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.show()

plot_decision_boundary(X, y, nn_model)