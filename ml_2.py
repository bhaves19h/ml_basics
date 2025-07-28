from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Load the dataset
iris = load_iris()
X = iris.data #sepal petal size (input)
y = iris.target #flower name  (output)

# Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)#size 0.2 mean 20% go for testing remaing 80 for data training
# random =1 beacuse Without it, the split will be different on every run

# Step 3: Create and train the model
model = KNeighborsClassifier(n_neighbors=3) #It compares new data points with existing training data.

#It checks which class is most common among those 3 and returns that.
#If among 3 neighbors:
#2 are setosa
#1 is virginica ✅ The model will predict: setosa

model.fit(X_train, y_train)

# Step 4: Predict for new data
sample = [[4.5, 1.0, 6.5, 3.0]]  # sepal & petal lengths/widths
prediction = model.predict(sample)

# Step 5: Output
print("Predicted flower class index:", prediction[0])
print("Flower name:", iris.target_names[prediction[0]])
