from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib as jb

# load the iris dataset
iris = load_iris()

# splitting our data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.33, shuffle=True)

# initializing the scikit learn KNN classifier
knn = KNeighborsClassifier(n_neighbors=7)

# fitting on train data
knn.fit(X_train, y_train)

# predicting on test data
predictions = knn.predict(X_test)

# metrics
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# saving model as a pickle/serialized file
jb.dump(knn, "models/iris_model.pkl")

print("Model Saved to models folder")
