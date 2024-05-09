import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("applicants.csv")

# Input sequence
print("Welcome to the applicant ranking system ML model")
print("Please input the prompted fields with desired applicant")
ce_y = int(input("How many years of preferred/required coding experience? "))
ce_l = input("Which coding languages are preferred/required? Please write in a comma separated list. (ex. Python, Java) ")
ed = input("What is the minimum degree requirement? (ex. High School Diploma, Associate's Degree, Bachelor's Degree, Master's Degree)")
m = input("What are the desired major(s)? Please write in a comma separated list. (ex. Computer Science, Biology)")
i = input("Is prior internship experience required? (Yes/No) ")
ie = int(input("How many years of desired inustry experience? "))

# Defining qualified column
def qualification_rating(cy, cl, ed, m, i, ie):
    ratings = []
    likelihood = []
    
    # Iterate through data frame rows
    for index, row in data.iterrows():
        curr_rating = 0
        llh = 0

        # Coding Experience (Years)
        if row["Coding Experience (Years)"] >= cy:
            curr_rating += row["Coding Experience (Years)"] - cy
        else:
            curr_rating -= 1

        # Coding Experience (Languages)
        lang = cl.split(", ")
        if isinstance(row["Coding Experience (Languages)"], str):
            if any(language in row["Coding Experience (Languages)"].split(", ") for language in lang):
                curr_rating += 1
            else:
                curr_rating -= 1
        else:
            # Handle cases where the value is not a string (e.g., empty or NaN)
            curr_rating -= 1


        # Education
        ed_vals = {"High School Diploma": 0, "Associate's Degree": 1, "Bachelor's Degree": 2, "Master's Degree": 3}
        if ed_vals.get(row["Education"], -1) >= ed_vals.get(ed, -1):
            curr_rating += ed_vals[row["Education"]] - ed_vals[ed] + 1

        # Major
        if row["Major"] in m.split(", "):
            curr_rating += 1
        else:
            curr_rating -= 1

        # Internship
        if (i == "No" and row["Internship"] == "Yes") or (i == "Yes" and row["Internship"] == "No"):
            curr_rating -= 1

        # Industry Experience
        if row["Industry Experience"] >= ie:
            curr_rating += row["Industry Experience"] - ie
        else:
            curr_rating -= 1

        ratings.append(curr_rating)

        # Likelihood
        if (curr_rating >= 4.0):
            llh = 1
        else:
            llh = 0

        likelihood.append(llh)

    return ratings, likelihood
 
data["Qualified"], data["Likelihood"] = qualification_rating(ce_y, ce_l, ed, m, i, ie)

# Sort candidates based on qualification ratings
top_candidates = data.sort_values(by='Qualified', ascending=False).head(20)

# Output the top 20 candidate's first/last name, and number
for index, candidate in top_candidates.iterrows():
    print(f"{candidate['First Name']} {candidate['Last Name']}: {candidate['Qualified']}")
    print(f"Likelihood: {candidate['Likelihood']}")
    print(f"Gender: {candidate['Gender']}")
    print(f"Ethnicity: {candidate['Ethnicity']}")
    print(f"Disability: {candidate['Disabilities']}")
    print(f"Veteran: {candidate['Veteran']}")
    print(f"Marital Status: {candidate['Marital Status']}")
    print()


# Select relevant features
X = data[['Coding Experience (Years)', 
          'Coding Experience (Languages)', 
          'Education', 
          'Major', 
          'Internship', 
          'Industry Experience',
          'Gender', 
          'Ethnicity', 
          'Disabilities', 
          'Veteran', 
          'Marital Status']]

y = data['Likelihood']

# Split the data into training, validation, and testing sets
train_data, test_and_validation_data = train_test_split(data, test_size=0.2, random_state=3)
validation_data, test_data = train_test_split(test_and_validation_data, test_size=0.5, random_state=3)

# # Likelihood Model
# likelihood_model = LogisticRegression(penalty='l2', C=1e23, random_state=1)
# likelihood_model.fit(train_data[X], train_data['Likelihood'])

# # Define preprocessing steps
# numeric_features = ['Coding Experience (Years)', 'Industry Experience']
# categorical_features = ['Gender', 'Ethnicity', 'Disabilities', 'Veteran', 'Marital Status', 'Education', 'Major', 'Internship']
# numeric_transformer = StandardScaler()
# categorical_transformer = OneHotEncoder()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])

# # Define the model
# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', LogisticRegression())])

# # Fit the model
# clf.fit(X_train, y_train)

# # Predict on the test set
# y_pred = clf.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

