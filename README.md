# Credit_Risk_Analysis
This script uses a Logistic Regression Model to identify the creditworthiness of borrowers using a dataset of historical lending activity from a peer-to-peer lending services company

## Overview of the Analysis

* __Purpose of Analysis:__ Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this script, we use various techniques to train and evaluate models with imbalanced classes. We use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. The purpose here is to classify incoming loans as either healthy or high risk to minimize the cost of defaulting loans.

* __Dataset:__ The dataset used ('../Resources/lending_data.csv') in the analysis contains all the information regarding the loan in question. It provides all the financial details and ratios like, loan size, interest rate, borrower income, debt to income ratio, number of user accounts, total debt etc. Alongwith all this financial information, one of the columns also provides the status of the loan as a binary number, where 0 means a healthy loan and 1 means a high risk loan.

* __Variables:__ The dataset provides info on 77536 loans out of which 75036 loans are healthy loans and 2500 loans are high risk. From this we can see that the dataset is highly imbalanced.

* __Stages of Machine Learning:__ After reading the dataset, it is first split into two sets, one for training the model and the other for testing the model. A logistic Regression Model is then created. Using this instance we fit the training data to the model and then use this trained model to predict values for our test data. The modelling is done on two sets of data, the original dataset and a resampled data. In the earlier step, we determined that the data is highly imbalanced, so we over sample the training data, so that we have equal amounts of both the outcomes for making accurate predictions. This over sampling is done using RandomOverSampler(). For classification, we use the Logistic Regression model, the instance of which is created by using LogisticRegression() function from the sci-kit learn package. We then evaluate both the models by calculating the accuracy score and generating the classification report using the confusion matrix.

Following are the steps in brief that are followed:

1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns. A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  

3. Check the balance of the labels variable (`y`) by using the `value_counts` function.

4. Split the data into training and testing datasets by using `train_test_split`.

5. Fit a logistic regression model by using the training data (`X_train` and `y_train`).

6. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

7. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.

 8. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

9. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

10. Evaluate the model’s performance by doing the following:

    * Calculate the accuracy score of the model.

    * Generate a confusion matrix.

    * Print the classification report.
---
## Results
<br>
* Machine Learning Model 1:
  * Logistic Regression Model that models an imbalanced dataset
  * Accuracy is 95.20%, i.e the model predicts outcomes with an accuracy of 95%
  * Precision for healthy loans = 100% (Model identifies Healthy loans 100% of times)
  * Recall value for healthy loans = 99% (Model classifies Healthy loans 99% of times)
  * Precision for high risk loans = 85% (Model identifies High Risk loans 85% of times)
  * Recall value for high risk loans = 91% (Model classifies High Risk loans 91% of times)
  <br>
* Machine Learning Model 2:
  * Logistic Regression Model that models an an oversampled dataset, where the high risk loans are oversampled to match the number of healthy loans
  * Accuracy is 99.36%, i.e the model predicts outcomes with an accuracy of 99.36%
  * Precision for healthy loans = 100% (Model identifies Healthy loans 100% of times)
  * Recall value for healthy loans = 99% (Model classifies Healthy loans 99% of times)
  * Precision for high risk loans = 84% (Model identifies High Risk loans 84% of times)
  * Recall value for high risk loans = 99% (Model classifies High Risk loans 99% of times)

---

## Summary

* From the results we can see that Model 2 performs better than MOdel 1, which is evident from the significant improvement in accuracy from 95% to 99%
* The precision and recall values for healthy loans remain the same in both the models. 
* We see a significant improvement in the recall value of high risk loans from 91% to 99%, which means that the resampled data model correctly classifies the high risk loans 99% of the time and only classifies 1% of the high risk loans as healthy loans, thus reducing the cost of a default loan significantly. 
* This improvement in recall value is achieved by sacrificing the precision value for high risk loans by 1%. This means that with the resampled data model, 16% of the high risk loans are identified as healthy instead of 15% with the original dataset. A further analysis of loans can be done to further analyze the loans classified as healthy loans, whic could potentially reduce this number too. This will just involve some overhead cost of extra analysis, which will be significantly lower than the cost of a default loan. 
 
 Taking all of the observations into account, the resampled data Logistic Regression model is highly recommended for the provided data set. 

 ## Contributors

Abhishika Fatehpuria (abhishika@gmail.com)

---

## License

MIT