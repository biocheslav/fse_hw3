#!/usr/bin/python3
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from useful_package import hyperbola


train_size, test_size = 1000, 100
X_train = 1 + np.random.rand(train_size).reshape(-1, 1)
Y_train = hyperbola(X_train)
X_test = 1 + np.random.rand(test_size).reshape(-1, 1)
Y_test = hyperbola(X_test)

model = RandomForestRegressor()
model.fit(X_train, Y_train)
mse = np.sum(np.abs(model.predict(X_test) - Y_test))
print(mse)
