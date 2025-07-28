from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample data: Hours studied vs Scores
hours = np.array([[1], [2], [3], [4], [5]])
scores = np.array([10, 20, 30, 40, 50])

# Step 1: Create the model
model = LinearRegression()

# Step 2: Train the model
model.fit(hours, scores)

# Extract slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

# Predict
predicted_score = model.predict([[6]])
print(f"Predicted score for 6 hours of study: {predicted_score[0]}")

# Given score = 60, calculate required hours
target_score = 60
required_hours = (target_score - b) / m

print(f"To score {target_score}, a student needs to study approximately {required_hours:.2f} hours.")

# Optional: Plotting
plt.scatter(hours, scores, color='blue')
plt.plot(hours, model.predict(hours), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours vs Score")
plt.show()
