import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

predictions = loaded_model.predict(X_test)

accuracy = loaded_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)
print(f"Predicted class: {iris.target_names[prediction[0]]}")