# FROM titanic dataset we are trying to find out the survival rate based on some dependant variable
import pandas as pd
#load the train and test data: This is part of preprocessing the dataset
train_data=pd.read_csv('/users/kal/titanic/train.csv')
test_data=pd.read_csv('/users/kal/titanic/test.csv')
#This command will print out some parts of the csv file to see if the data is loaded properly
print(train_data.head())
# handle missing values: take the median age and fill in where the age is blank
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
#missing embarked values Replace missing values in Embarked with the mode (most common value).
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
#Drop the Cabin Column: Since Cabin has too many missing values, it’s often best to drop it.
train_data.drop(columns=['Cabin'], inplace=True)
test_data.drop(columns=['Cabin'], inplace=True)
#Convert categorical variables like Sex and Embarked into numerical representations using label encoding or one-hot encoding.
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
#One-Hot Encode Embarked so embarked will be categorized into 1 or 0 and redudant variables will be dropped
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)
#feature engineering, you can add new variables to make the data more meaningful for ML
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']
train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 0).astype(int)
#train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 0).astype(int)
#Scaling features like Age and Fare can improve the performance of certain models.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[['Age', 'Fare']] = scaler.fit_transform(train_data[['Age', 'Fare']])
test_data[['Age', 'Fare']] = scaler.transform(test_data[['Age', 'Fare']])
#Remove columns that don’t contribute to the prediction, like PassengerId, Name, and Ticket.
train_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
test_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)


#Separate the target variable (Survived) from the features in the training set.

X_train = train_data.drop(columns='Survived')
y_train = train_data['Survived']
X_test = test_data  # Note: test data doesn’t have the Survived column

#logistic regression good for linear data to train/fit the model. Here it will look at probability values and classify it as 1 or 0.
# if the probability of survival is more than 0.5 it will say 1-survived or 0- didn't survive
#it basically assigns weights/coeffciients to the features x_train to predict y_pred
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_reg.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train, y_pred)}")

#accuracy was 80% so we can use another model decision tree
#decision tree Captures non-linear relationships, interpretable, can handle missing values (although we’ve filled them).
#rProne to overfitting on smaller datasets unless tree depth is restricted.
from sklearn.tree import DecisionTreeClassifier

# Initialize the model with a max depth to prevent overfitting
tree = DecisionTreeClassifier(max_depth=3)

# Train and predict
tree.fit(X_train, y_train)
y_pred = tree.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train, y_pred)}")
#accuracy is 82% so it has improved a bit. Lets use random forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train and predict
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train, y_pred)}")

#lets try svm
from sklearn.svm import SVC

# Initialize the model
svm = SVC(kernel='linear', C=1)

# Train and predict
svm.fit(X_train, y_train)
y_pred = svm.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train, y_pred)}")

#svm gave a smaller prediction 79%

from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train and predict
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)
print(f"Accuracy: {accuracy_score(y_train, y_pred)}")
#this also gave a high perdiction 85%

from sklearn.model_selection import cross_val_score

# Since random forest is giving the best data, I can choose random forest and do cross validation.
# Evaluate with cross-validation. I split the data into multiple data and test it multiple times on random forest. 
# in this case cv=5 means the dataset is split into 5 parts. imes (each part gets a turn being the test set).
 #after each teration, the model’s performance (accuracy in this case) is recorde.the final cross-validation score is the average accuracy across all 5 iterations, providing a more robust estimate of model performance. 

rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {rf_scores.mean()}")