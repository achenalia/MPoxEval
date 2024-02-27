# AI Monkey-Pox Evaluation Project

## Overview
Welcome to the AI Monkey-Pox Evaluation Project! This project utilizes an Artificial Neural Network (ANN) to assess the likelihood of Monkey-Pox based on a set of symptoms and user input. The ANN has been trained on anonymized patient data to provide accurate predictions.

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
- DO NOT USE THIS AS YOUR ONLY SOURCE OF MEDICAL ADVICE, SEEK A QUALIFIED HEALTHCARE PROFESSIONAL IF YOU ARE CONCERNED.

Feel free to explore and utilize this AI Monkey-Pox Evaluation Project for informative insights into potential health concerns. Stay informed and proactive about your well-being!
