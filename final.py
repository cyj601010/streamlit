# %%
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# %%

s=pd.read_csv("https://raw.githubusercontent.com/cyj601010/streamlit/main/social_media_usage.csv")
print(s.head())

# %%
st.markdown("# Welcome to my streamlit app!")

st.markdown("# 

# %%
print(s.dtypes)

# %%
# Check the total number of missing values in each column
missing_values = s.isnull().sum()

# Print the result
print("Total number of missing values in each column:")
print(missing_values)

# %%
num_rows, num_columns = s.shape
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# %% [markdown]
# ---

# %% [markdown]
# ## 2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected.

# %%
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a toy DataFrame
df = {'Column1': [1, 2, 0],
        'Column2': [1, 0, 1]}
toy = pd.DataFrame(df)

# Apply the clean_sm function to each element in the DataFrame
cleaned_df = toy.apply(clean_sm)


# %%

# Display the original and cleaned DataFrames
print("Original DataFrame:")
print(toy)
print("\nCleaned DataFrame:")
print(cleaned_df)

# %% [markdown]
# ---

# %% [markdown]
# ## 3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
# 

# %%
# Step 1: Create a new dataframe "ss"
ss = pd.DataFrame(s)
ss.head()

# %%
# Step 2: Create the target column "sm_li"
ss['sm_li'] = s['web1h'].apply(clean_sm)
ss.head()

# %%
# Step 3: Add features to the dataframe

# Income Feature
def clean_in(x):
    return x if (1 <= x <= 9) else np.nan
ss['incomen'] = s['income'].apply(clean_in)
ss.head()

#def clean_in(x):
#    return np.where((x >= 1) & (x <= 9), 1, np.nan)
#ss['incomen'] = s['income'].apply(clean_in)
#ss.head()

# %%
# Education Feature
def clean_ed(x):
    return x if (1 <= x <= 8) else np.nan
ss['educ2n'] = s['educ2'].apply(clean_ed)
ss.head()

# %%
# Parent Feature
def clean_pa(x):
    return np.where(x == 1, 1, 0)
ss['parn'] = s['par'].apply(clean_pa)
ss.head()

# %%
# Married Feature
def clean_ma(x):
    return np.where(x == 1, 1, 0)
ss['maritaln'] = s['marital'].apply(clean_ma)
ss.head()

# %%
# Gender Feature
def clean_ge(x):
    return np.where(x == 2, 1, 0)
ss['gendern'] = s['gender'].apply(clean_ge)
ss.head()

# %%
#Age Feature
def clean_ag(x):
    return x if (1 <= x <= 98) else np.nan
ss['agen'] = s['age'].apply(clean_ag)
ss.head()

# %%
# Check the total number of missing values in each column
missing_values = ss.isnull().sum()

# Print the result
print("Total number of missing values in each column:")
print(missing_values)

# %%
# Assuming you have a function clean_sm to create the target variable
#def clean_sm(value):
#    return 1 if value == 1 else 0

# %%
# Step : Drop missing values
ss = ss.dropna()

# %%
# Check the total number of missing values in each column
missing_values = ss.isnull().sum()

# Print the result
print("Total number of missing values in each column:")
print(missing_values)
print(s.dtypes)

# %%
# Step 5: Exploratory analysis
#grouped_ss = ss.groupby('sm_li').describe()

# Display summary statistics
#print(grouped_ss)
columns_to_plot = ['incomen', 'educ2n', 'maritaln', 'gendern', 'agen']
# Loop through columns and create bar plots grouped by 'sm_li'
for column in columns_to_plot:
    plt.figure()
    sns.countplot(data=ss, x=column, hue='sm_li')
    plt.title(f'Bar Plot of {column} grouped by sm_li')
    plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ## 4. Create a target vector (y) and feature set (X).

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming you have a DataFrame called df with your data
# Replace the column names and data with your actual dataset

# Step : Create a target vector (y) and feature set (X)
X = ss[['incomen', 'educ2n', 'maritaln', 'gendern', 'agen']]
y = ss['sm_li']


# %% [markdown]
# ---

# %% [markdown]
# ## 5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# %%
# Step : Split the data into training and test sets (hold out 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# #### This step splits the all observations into 80% training sets and 20% into test sets. We will build model using the 80% training sets so that we can predict for the future. Then, we will test the model to the 20% of test sets to see if our model predicts well.

# %% [markdown]
# ---

# %% [markdown]
# ## 6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# %%
# Step 6 : Instantiate a logistic regression model and fit it with the training data
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# %% [markdown]
# ---

# %% [markdown]
# ## 7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents.

# %%
# Step 7: Evaluate the model using the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')


# %% [markdown]
# ---

# %% [markdown]
# ## 8.Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# %%
# Step 8: Generate a confusion matrix and interpret it
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# %% [markdown]
# #### The left sides are the negatives, and the right sides are the positives. Meaning, there were 99 true negatives, 62 false positives, 24 false negatives, 67 true positives. It appears the true values are somewhat higher than the false values. Thus, we can conclude that the model is working. However, the model seems to have a very low accuracy.

# %%
# Step: Create a confusion matrix DataFrame with informative column and index names
conf_matrix_df = pd.DataFrame(conf_matrix, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1'])
print('Confusion Matrix DataFrame:')
print(conf_matrix_df)

# Step: Calculate precision, recall, and F1 score by hand
precision = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')

# Step : Create a classification report using sklearn
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:')
print(classification_rep)

# %% [markdown]
# #### Precision measures the accuracy of the positive predictions made by the model. Recall measures the ability of the model to capture all positive instances. F1 Score is the balanced mean of precision and recall. In our case, we would like to learn the demographics of the Linked in users. Thus, having real data is imporant. Thus, it is important that we have a good recall value.

# %% [markdown]
# ---

# %% [markdown]
# ## 9. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?  

# %%
# Step : Use the model to make predictions for specific scenarios
# Replace the values with your desired input
#'incomen', 'educ2n', 'maritaln', 'gendern', 'agen'
example_high_In_Edu_marital = [[8, 7, 1, 1, 42]]

probability1 = model.predict_proba(example_high_In_Edu_marital)[:, 1][0]

print(f'Probability for the first example: {probability1:.2%}')

")


