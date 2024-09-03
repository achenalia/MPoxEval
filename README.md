# AI Monkey-Pox Evaluation Project (v1.0.0, re-work coming soon!)

## Overview
Welcome to the AI Monkey-Pox Evaluation Project! This project is a local CLI tool which utilizes an Artificial Neural Network to assess symptoms and provide recommendations on Monkey-Pox testing through user input and machine learning to make predictions regarding the patient's monkey-pox risk and whether to seek medical attention based on their responses

## Dataset
The dataset used in this project contains information on various symptoms related to Monkey-Pox, such as Rectal Pain, Sore Throat, Oedema below the Waist, Oral Lesions, Solitary Lesion, Swollen Tonsils, HIV Infection, and Sexually Transmitted Infection.

## Model Training
The project involves encoding categorical data using LabelEncoder, splitting the data into training and testing sets, and building a neural network model using Keras. The model is trained on the training set with specified epochs and batch size to optimize accuracy.

## User Interaction
Users are prompted with questions related to symptoms and systemic illness. Based on their responses, the ANN generates predictions to recommend whether testing for Monkey-Pox is necessary.

## Instructions
1. Run the provided code to train the model and set up the user interaction.
2. Answer the questions prompted regarding symptoms and systemic illness.
3. Receive a recommendation based on the input provided by the ANN.

## Recommendations
- If the prediction probability is above 0.6, testing for Monkey-Pox is recommended.
- If all symptom responses are 'No' and systemic illness is not present, further evaluation by a healthcare professional is advised.
- For other scenarios, seeking medical advice is recommended based on the symptoms reported.
- SEEK MEDICAL ATTENTION IF CONCERNED, DO NOT USE THIS AS ONLY SOURCE OF MEDICAL ADVICE.

## Setting Up Environment

### Activating Virtual Environment
To activate the .venv virtual environment:
- On Windows:
  ```
  .venv\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source .venv/bin/activate
  ```

### Installing Dependencies from requirements.txt
Once the virtual environment is activated, install dependencies using pip:
```
pip install -r requirements.txt
```

By following these steps, you ensure a consistent development environment with all necessary dependencies installed for running the AI Monkey-Pox Evaluation Project. Stay informed and proactive about your health with this innovative tool!
