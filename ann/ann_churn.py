import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

# Encoding the categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pdb
pdb.set_trace()
encoder_X1 = LabelEncoder()
X[:, 1] = encoder_X1.fit_transform(X[:, 1])
encoder_X2 = LabelEncoder()
X[:, 2] = encoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Avoid dummy variable trap(multi-collinearity error)

# Train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Perform feature scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Part 2- Make the ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout

# # initialising the ANN

# model = Sequential() # Sequential model

# # Adding input and hidden layer
# # use relu for input and hidden layers, sigmoid for output layer

# model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11)) # units => specify units for output i.e hidden layer
#                                                                                   # init => initialize weights to small values
# model.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu')) # Hiddden layer units

# model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid')) # Use sigmoid if your classes are binary. Use softmax otherwise

# # Compile the model

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # Loss function:- binary cross used when output var is binary. For more than 2 classes, use categorical crossentropy

# model.fit(X_train, y_train, epochs=100, batch_size=10)

# [error, acc] = model.evaluate(X_test, y_test)

# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)
# from sklearn.metrics import confusion_matrix
# matrix = confusion_matrix(y_test, y_pred)
# print(matrix)


# Improving the ANN
from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer='rmsprop', kernel_initializer='glorot_uniform', p=0.1):
    """
    Function to generate layers for the ANN
    """
    model = Sequential()
    model.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    model.add(Dropout(rate=p))
    model.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
    model.add(Dropout(rate=p))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=build_classifier)
kernel_initializer = ['glorot_uniform', 'normal', 'uniform']
optimizers = ['rmsprop', 'adam']
epochs = [50, 100, 150]
batches = [10, 20, 50]
p_values = [0.1, 0.2, 0.3]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, kernel_initializer=kernel_initializer, p=p_values)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
