
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load the data
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Hackathon\\transactions2.csv")
data['isFraud'] = data['isFraud'].replace({0: 'Not Fraud', 1: 'Fraud'})
# Specify the columns to be deleted




# # print(data.head())


# print(data.info())
# data.info()
# #Building Models
# #Model 1
# #SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# X = pd.get_dummies(data.drop(['isFraud'], axis=1))
# Define the features (X) and the target variable (y)
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Initialize the SVM model
svm = SVC()

# Fit the SVM model on the training data
svm.fit(X_train, y_train)

# Predict on the test data
predictions1 = svm.predict(X_test)

# Evaluate the model
accuracy1 = accuracy_score(y_test, predictions1)
print("Accuracy of SVM:", accuracy1)
#Accuracy for this model is 0.9951612903225806
# #Naive Bayes
# from sklearn.naive_bayes import GaussianNB
# # Initialize the Naive Bayes model
# nb = GaussianNB()

# # Fit the Naive Bayes model on the training data
# nb.fit(X_train, y_train)

# # Predict on the test data
# predictions2 = nb.predict(X_test)

# # # Evaluate the model
# accuracy2 = accuracy_score(y_test, predictions2)
# print("\nAccuracy of Naive Bayes:", accuracy2)
# # Accuracy for this model is 0.5951612903225807

# #Decision Tree
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()

# # Fit the Decision Tree model on the training data
# dt.fit(X_train, y_train)

# # Predict on the test data
# predictions3 = dt.predict(X_test)

# # Evaluate the model
# accuracy3 = accuracy_score(y_test, predictions3)
# print("\nAccuracy of Decision Tree:", accuracy3)
# # Accuracy for this model is 0.9954838709677419

# #Random Forest
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()

# # Fit the Random Forest model on the training data
# rf.fit(X_train, y_train)

# # Predict on the test data
# predictions4 = rf.predict(X_test)

# # Evaluate the model
# accuracy4 = accuracy_score(y_test, predictions4)
# print("\nAccuracy of Random Foresr\t:", accuracy4)
# # # Accuracy for this model is 0.9954838709677419

# # Logistic Regression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()

# # Fit the model on the training data
# lr.fit(X_train, y_train)

# # Make predictions on the test data
# predictions = lr.predict(X_test)

# # Calculate the accuracy of the model
# accuracy5 = accuracy_score(y_test, predictions)
# print("\nAccuracy of logistic regression is:", accuracy5)
# # Accuracy for this model is 0.9987096774193548

 # From This all algorithms we can see that Logistic Regression has the highesy accuracy. Hence we will use logistic regression for predicting the fake transactions.


import joblib
# Save the trained model to a file
joblib.dump(svm, 'transactions_model2.pkl')




