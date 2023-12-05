import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
# A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - ESME RICHARDSON

# Preprocessing used: Since this file contains data such as Patient ID, I have ensured that this category is not
# considered in the program. Once irrelevant data is found, it is important to only encode the attributes necessary
# for the program to function, excluding those deemed not useful. It is also important to consider data which may not
# fit conventions needed to process it, such as values other than TRUE/FALSE for boolean-expecting questions such as
# those for the symptoms. If any mismatched answers are found, these are then excluded from the data used. It is also
# important to look for null/empty data, which may cause problems down the line, and exclude those as well. After
# these steps are taken to normalize the data, it is safe to proceed with the processing and pass it to the program.

# Read the dataset provided and outline the different categories necessary:
data = pd.read_csv("MPox_Dataset.csv")
features = ['Rectal Pain', 'Sore Throat', 'Oedema below the Waist', 'Oral Lesions',
            'Solitary Lesion', 'Swollen Tonsils', 'HIV Infection', 'Sexually Transmitted Infection']
target = 'MonkeyPox'

# Encode category data as necessary:
encoder = LabelEncoder()
for feature in features:
    data[feature] = encoder.fit_transform(data[feature])

# Encode target variable 'MonkeyPox' with LabelEncoder
y = encoder.fit_transform(data[target])

# Create training and testing sets:
X = data[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Establish the neural network we will use:
model = Sequential()
model.add(Dense(20, input_dim=len(features), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up the model for training:
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Set up print statements to welcome the user, explain the system, and display the accuracy of the ANN:
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy of the ANN: {accuracy * 100:.4f}")
print("Welcome to the MPox Test Recommender System -- an Artificial Neural Network (ANN) trained on anonymized "
      "patient data.")
print(f"This ANN has an accuracy of {accuracy * 100:.4f}.")

# Ask 'Systemic Illness' question separately:
print("Are you experiencing symptoms typical of a Systemic Illness such as Fever, Fatigue, Swollen Lymph Nodes, "
      "or Muscle Weakness? (Yes/No): ")
systemic_illness = input().strip().lower()
while systemic_illness not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    systemic_illness = input().strip().lower()

# Init user_input storage for answers:
user_input = {}

# Print beginning instruction:
print("\nAnswer the following questions:")
# Dictionary of questions to be asked:
questions = {
    'Rectal Pain': "Are you suffering from Rectal Pain? (Yes/No): ",
    'Sore Throat': "Are you suffering from a Sore Throat? (Yes/No): ",
    'Oedema below the Waist': "Are you experiencing Oedema (swelling) below the Waist? (Yes/No): ",
    'Oral Lesions': "Are you suffering from Oral Lesions (ie. sores)? (Yes/No): ",
    'Solitary Lesion': "Are you suffering from Solitary Lesions (ie. sores) anywhere on your body? (Yes/No): ",
    'Swollen Tonsils': "Are you suffering from Swollen Tonsils? (Yes/No): ",
    'HIV Infection': "Has your physician diagnosed you with an HIV Infection? (Yes/No): ",
    'Sexually Transmitted Infection': "Has your physician diagnosed you with a Sexually Transmitted Infection (STI)? "
                                      "(Yes/No):"
}

# Ask the questions in the dictionary and encode the user's input:
for feature, question in questions.items():
    answer = input(question).strip().lower()
    while answer not in ['yes', 'no']:
        print("Please enter a valid response: Yes or No.")
        answer = input(question).strip().lower()
    user_input[feature] = answer

# Transform user input into DataFrame for prediction:
user_data = pd.DataFrame([user_input])

# Create dummy variables for user input:
user_data = pd.get_dummies(user_data)

# Ensure user input contains all features by aligning columns:
missing_cols = set(X.columns) - set(user_data.columns)
for col in missing_cols:
    user_data[col] = 0

user_data = user_data[X.columns]

# Use the ANN to create predictions to output:
prediction = model.predict(user_data)

# Print predictions to user:
if prediction >= 0.6:
    print("\nBased on the data you provided, you are recommended to get tested for Monkey-Pox as soon as possible.")
elif all(value == 'no' for value in user_input.values()) and systemic_illness == 'no':
    print("\nYou answered 'No' to all symptoms. This means you may not need to be tested for Monkey-Pox, but it is "
          "recommended to consult a healthcare professional for further "
          "evaluation if you deem it necessary.")
else:
    print("\nBased on the data you provided, it does not appear that you need to be tested for Monkey-Pox. However, "
          "due to your symptoms, it is recommended to seek care from a healthcare professional at your earliest "
          "convenience.")
