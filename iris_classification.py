import joblib
import numpy as np
#will need it later on to create a new instance in the form of numbay array
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
#imported needed libraries


#loading  dataset and saving target variables and features
iris = load_iris()
X, y = iris.data, iris.target

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# writing the model
k = 3  # Nbr of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

#making prediction
y_pred = knn.predict(X_test)


#evaluationg the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
#extra (visualisation , testing with an instance , saving and loading the model)




# Visualization of classes based on Sepal Length and Petal Length
plt.figure(figsize=(10, 6))

# Color map and class names for consistent visualization
colors = ['red', 'green', 'blue']
class_names = iris.target_names

# Plot each class with a different color
for i, color in enumerate(colors):
    # Select data points for this class
    class_mask = (y == i)
    plt.scatter(
        X[class_mask, 0],  # Sepal Length (first feature)
        X[class_mask, 2],  # Petal Length (third feature)
        # we chosen just two features from the four we have so the visualization
        # can be simple (bi-dimensional graph)
        c=color,
        label=class_names[i],
        edgecolor='black',
        linewidth=1,
        alpha=0.7
    )

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Iris Dataset: Sepal Length vs Petal Length')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#sqving the model :
"""
# Save the model
model_path = 'models/iris_knn_model.joblib'
joblib.dump(knn, model_path)
print(f"Model saved to {model_path}")
"""

#loading the model :
"""
# Load the model
loaded_knn = joblib.load(model_path)
"""
# Create a new instance to test
"""
new_instance = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example instance with sepal length 5.1, sepal width 3.5, etc.
"""
# Predict the class of the new instance
"""
predicted_class = loaded_knn.predict(new_instance)
"""
# Plot the new instance
"""
plt.scatter(
    new_instance[0, 0],  # Sepal Length
    new_instance[0, 2],  # Petal Length
    c='purple',  # Distinctive color for the new instance
    marker='x',  # Different marker to stand out
    s=200,  # Larger size
    label='New Instance',
    edgecolor='black',
    linewidth=2
)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Iris Dataset: Sepal Length vs Petal Length\nwith New Instance Prediction')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Print the prediction for the new instance
print(f'Predicted class for new instance: {class_names[predicted_class[0]]}')

plt.show()
"""