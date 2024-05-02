import math
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

sns.set()
%matplotlib inline

applicants = pd.read_csv('applicants.csv')

train_data, test_and_validation_data = train_test_split(applicants, test_size=0.2, random_state=3)
validation_data, test_data = train_test_split(test_and_validation_data, test_size=0.5, random_state=3)

applicants['rating'] = pd.Series(ratings)
ratings = []
for index, row in applicants.iterrows():
    # Initialize rating for the current applicant
    rating = 0
    
    # Apply conditions to assign ratings based on different criteria
    if row['coding_years'] > 2:
        rating += 1
    # Add more conditions as needed for other features
    # For example:
    # if row['coding_language'] == 'Python':
    #     rating += 1
    # if row['education'] == 'Bachelor':
    #     rating += 1
    # Add more conditions for other features
    
    # Append the rating to the list of ratings
    ratings.append(rating)

applicants['suitable'] = applicants['rating'].apply(lambda rating : +1 if rating > 3 else -1)

features = ['coding_years', 
            'coding_language',
            'education',
            'major',
            'internship',
            'industry_experience']

target = 'suitable'

evaluation_model = LogisticRegression(penalty='l2', C=1e23, random_state=1)
evaluation_model.fit(train_data[features], train_data[target])

y_pred = evaluation_model.predict(validation_data[features])
accuracy = accuracy_score(validation_data[target], y_pred)



dict = {}
for index, row in applicants.iterrows():
    employee_id = row['employee_id']
    address = row['address']
    gender = row['gender']
    ethnicity = row['ethnicity']
    disabilities = row['disabilities']
    veteran = row['verteran']
    marital_status = row['marital_status']

    tuple = (address, address, gender, ethnicity, disabilities, veteran, marital_status)

    dict[employee_id] = tuple
