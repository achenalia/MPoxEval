import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Read the dataset provided and outline the different categories necessary:
data = pd.read_csv("MPox_Dataset.csv")
features = ['Systemic Illness', 'Rectal Pain', 'Sore Throat', 'Oedema below the Waist', 'Oral Lesions',
            'Solitary Lesion', 'Swollen Tonsils', 'HIV Infection', 'Sexually Transmitted Infection']
target = 'MonkeyPox'

# Encode category data as necessary:
encoder = LabelEncoder()
for feature in features:
    data[feature] = encoder.fit_transform(data[feature])

# Create training and testing sets:
X = data[features]
y = data[target]
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
print("What type of Systemic Illness are you experiencing?")
systemic_illness = input().strip().lower()

while systemic_illness not in ['fever', 'muscle aches and pain', 'swollen lymph nodes']:
    print("Please enter a valid Systemic Illness: Fever, Muscle Aches and Pain, or Swollen Lymph Nodes.")
    systemic_illness = input().strip().lower()

# Init the question array to be asked:
questions = [
    "Are you experiencing Rectal Pain? Provide your Answer as Yes or No.",
    "Are you experiencing Sore Throat? Provide your Answer as Yes or No.",
    "Are you experiencing any Oedema (swelling) below the waist? Provide your Answer as Yes or No.",
    "Do you have any Oral Lesions (ie. sores)? Provide your Answer as Yes or No.",
    "Do you have any Solitary Lesions (ie. sores) in any part of your body? Provide your Answer as Yes or No.",
    "Do you have swollen tonsils? Provide your Answer as Yes or No.",
    "Have you been diagnosed with an HIV Infection by your physician? Provide your Answer as Yes or No.",
    "Have you been diagnosed with any Sexually Transmitted Infections (STIs) by your physician? Provide your Answer "
    "as Yes or No."
]

# Set up the algorithm which will allow for answers from the user:
answers = []
for question in questions:
    answer = input(question).strip().lower()
    while answer not in ['yes', 'no']:
        print("Please enter a valid response: Yes or No.")
        answer = input(question).strip().lower()
    answers.append(answer)

# Define mappings for categorical values
categorical_mappings = {
    'fever': 0,
    'muscle aches and pain': 1,
    'swollen lymph nodes': 2,
    'yes': 1,
    'no': 0
}

# Consolidate the answers to use for prediction later:
user_input = {
    'Systemic Illness': systemic_illness,
    'Rectal Pain': answers[0],
    'Sore Throat': answers[1],
    'Oedema below the Waist': answers[2],
    'Oral Lesions': answers[3],
    'Solitary Lesion': answers[4],
    'Swollen Tonsils': answers[5],
    'HIV Infection': answers[6],
    'Sexually Transmitted Infection': answers[7]
}

# Map categorical inputs to encoded values
for feature in features:
    user_input[feature] = categorical_mappings.get(user_input[feature], user_input[feature])

# Use the ANN to create predictions to output:
user_data = pd.DataFrame([user_input])
prediction = model.predict(user_data[features])

# Print predictions to user:
if prediction >= 0.6:
    print("Based on the data you provided, you are recommended to get tested for Monkey-Pox as soon as possible.")
else:
    print("Based on the data you provided, it does not appear that you need to be tested for Monkey-Pox, however, "
          "due to your symptoms, you are recommended to seek care from a healthcare professional at your earliest "
          "convenience.")
