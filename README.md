# AI/ML Projects
Titanic project-

Dataset downloaded from Kaggle

Trying to predict the label-survival rate based on dependent variables/features like age, class, sex etc.

Language: Python, IDE-VBStudio

1. Pre processed Data 

Added missing values for age using median and mode

Dropped cabin column because it had too many missing values

Converted categorical values like sex and embarked to numerical representation(1 or 0) using label encoding or one hot encoding

2. Feature engineering

Added new variables/features to make the data more meaningful for ML
In this case family(parents and siblings) and Is_alone column
If family>0 then is_alone is False else true

3. Train/Fit the data
Train data-80% and test data-20%
X_Train: This set will have all the features/dependent variables
Y_Train: This set will have all the output/label

Y_Pred will have the output we are trying to predict from the training data set

X_train = train_data.drop(columns='Survived')
y_train = train_data['Survived']
X_test = test_data  # Note: test data doesn’t have the Survived column

4. Scaling features like Age and Fare can improve the performance of certain models.

5. ML selection
I used Logistic regression, Decision trees, random forest and SVM.
Random forest gave me the most accurate results-82% probability between y_pred and y_train

6. Cross validation

choose random forest and did cross validation 

# Evaluate with cross-validation. I split the data into multiple data and test it multiple times on random forest. 
# in this case cv=5 means the dataset is split into 5 parts. imes (each part gets a turn being the test set).

 #after each teration, the model’s performance (accuracy in this case) is recorde.the final cross-validation score is the average accuracy across all 5 iterations, providing a more robust estimate of model performance. 

rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {rf_scores.mean()}")


Created a GENAI custom Chatbot using OpenAI: 

Using React for front end
MSSQL for Database
OpenAI for API Calls



