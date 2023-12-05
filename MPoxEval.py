import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

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

# Ask 'Systemic Illness' question separately
print("Are you experiencing symptoms typical of a Systemic Illness such as Fever, Fatigue, Swollen Lymph Nodes, "
      "or Muscle Weakness? (Yes/No): ")
systemic_illness = input().strip().lower()
while systemic_illness not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    systemic_illness = input().strip().lower()

# Initialize user input dictionary
user_input = {}

# Ask each feature's question separately and gather user responses
print("\nAnswer the following questions:")

# Ask 'Rectal Pain' question
user_input['Rectal Pain'] = input("Are you suffering from Rectal Pain? (Yes/No): ").strip().lower()
while user_input['Rectal Pain'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Rectal Pain'] = input("Are you suffering from Rectal Pain? (Yes/No): ").strip().lower()

# Ask 'Sore Throat' question
user_input['Sore Throat'] = input("Are you suffering from a Sore Throat? (Yes/No): ").strip().lower()
while user_input['Sore Throat'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Sore Throat'] = input("Are you suffering from a Sore Throat? (Yes/No): ").strip().lower()

# Ask 'Oedema below the Waist' question
user_input['Oedema below the Waist'] = input("Are you experiencing Oedema (swelling) below the Waist? (Yes/No): ").strip().lower()
while user_input['Oedema below the Waist'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Oedema below the Waist'] = input("Are you experiencing Oedema (swelling) below the Waist? (Yes/No): ").strip().lower()

# Ask 'Oral Lesions' question
user_input['Oral Lesions'] = input("Are you suffering from Oral Lesions (ie. sores)? (Yes/No): ").strip().lower()
while user_input['Oral Lesions'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Oral Lesions'] = input("Are you suffering from Oral Lesions (ie. sores)? (Yes/No): ").strip().lower()

# Ask 'Solitary Lesion' question
user_input['Solitary Lesion'] = input("Are you suffering from Solitary Lesions (ie. sores) anywhere on your body? ("
                                      "Yes/No): ").strip().lower()
while user_input['Solitary Lesion'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Solitary Lesion'] = input("Are you suffering from Solitary Lesions (ie. sores) anywhere on your body? "
                                          "(Yes/No): ").strip().lower()

# Ask 'Swollen Tonsils' question
user_input['Swollen Tonsils'] = input("Are you suffering from Swollen Tonsils? (Yes/No): ").strip().lower()
while user_input['Swollen Tonsils'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Swollen Tonsils'] = input("Are you suffering from Swollen Tonsils? (Yes/No): ").strip().lower()

# Ask 'HIV Infection' question
user_input['HIV Infection'] = input("Has your physician diagnosed you with an HIV Infection? (Yes/No): ").strip().lower()
while user_input['HIV Infection'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['HIV Infection'] = input("Has your physician diagnosed you with an HIV Infection? (Yes/No): ").strip().lower()

# Ask 'Sexually Transmitted Infection' question
user_input['Sexually Transmitted Infection'] = input("Has your physician diagnosed you with a Sexually Transmitted "
                                                     "Infection (STI)? (Yes/No): ").strip().lower()
while user_input['Sexually Transmitted Infection'] not in ['yes', 'no']:
    print("Please enter a valid response: Yes or No.")
    user_input['Sexually Transmitted Infection'] = input("Has your physician diagnosed you with a Sexually Transmitted "
                                                         "Infection (STI)? (Yes/No): ").strip().lower()

# Transform user input into DataFrame for prediction:
user_data = pd.DataFrame([user_input])

# Use the ANN to create predictions to output:
prediction = model.predict(user_data[features])

# Print predictions to user:
if prediction >= 0.6 or systemic_illness == 'yes':
    print("\nBased on the data you provided, you are recommended to get tested for Monkey-Pox as soon as possible.")
elif all(value == 'no' for value in user_input.values()):
    print("\nYou answered 'No' to all symptoms. It is recommended to consult a healthcare professional for evaluation.")
else:
    print("\nBased on the data you provided, it does not appear that you need to be tested for Monkey-Pox. However, "
          "due to your symptoms, it is recommended to seek care from a healthcare professional at your earliest "
          "convenience.")
