# Import pandas module as pd
import pandas as pd

# Load the CSV files using pandas' read_csv function
dataset = pd.read_csv("dataset.csv", header=None)
symptom_precaution = pd.read_csv("symptom_precaution.csv", header=None)
symptom_severity = pd.read_csv("symptom_severity.csv", header=None)
disease_description = pd.read_csv("disease_description.csv", header=None)

# Initialize dictionaries for storing data
disease_symptom_data = {}
symptom_precaution_data = {}
symptom_severity_data = dict(symptom_severity.values.tolist())
disease_description_data = dict(disease_description.values.tolist())

# Populate disease_symptom_data dictionary
for index, row in dataset.iterrows():
    disease = row[0]
    symptoms = [symptom.strip() for symptom in row[1:].dropna().tolist()]
    disease_symptom_data[disease] = disease_symptom_data.get(disease, set()).union(set(symptoms))

# Populate symptom_precaution_data dictionary
for index, row in symptom_precaution.iterrows():
    disease = row[0]
    precautions = row[1:].dropna().tolist()
    symptom_precaution_data[disease] = precautions

# Heuristic function
def heuristic(user_symptoms, disease_symptoms):
    common_symptoms = user_symptoms.intersection(disease_symptoms)
    return (len(common_symptoms) / len(user_symptoms) if user_symptoms else 0)

# A* search algorithm
def a_star_search(user_symptoms, disease_data, symptom_severity_data, threshold=0.9):
    potential_diagnoses = []
    for disease, symptoms in disease_data.items():
        disease_symptoms = set(symptoms)
        h = heuristic(user_symptoms, disease_symptoms)
        if 0.6 < h < threshold:
            max_severity = max([symptom_severity_data.get(symptom, 0) for symptom in disease_symptoms], default=0)
            
            potential_diagnoses.append((disease, h, max_severity))
    return sorted(potential_diagnoses, key=lambda x: (x[1], -x[2]), reverse=True)

# Function to suggest a new symptom
def suggest_symptom(user_symptoms, disease_data, symptom_severity_data, all_asked_symptoms):
    symptom_counts = {}
    for _, symptoms in disease_data.items():
        common_symptoms = user_symptoms.intersection(symptoms)
        if common_symptoms:
            remaining_symptoms = symptoms.difference(user_symptoms).difference(all_asked_symptoms)
            for symptom in remaining_symptoms:
                if symptom in symptom_severity_data:
                    if symptom not in symptom_counts:
                        symptom_counts[symptom] = 0
                    symptom_counts[symptom] += len(common_symptoms)
    sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: (-x[1], symptom_severity_data.get(x[0], 0)), reverse=True)
    return sorted_symptoms[0][0] if sorted_symptoms else None

# Function to print diagnoses
def print_diagnoses(diagnoses, disease_description_data, symptom_precaution_data):
    print("\nPotential diagnoses:")
    for diagnosis, similarity, severity in diagnoses:
        print(f"{diagnosis} (similarity: {similarity * 100:.2f}%, severity: {severity})")
        print(f"Description: {disease_description_data.get(diagnosis, 'No description available')}")
        print("Precautions:")
        for precaution in symptom_precaution_data.get(diagnosis, ['No precautions available']):
            print(f"- {precaution}")
        print()

# Main chatbot function
def chatbot(disease_symptom_data, symptom_precaution_data, symptom_severity_data, disease_description_data):
    print("Welcome to the Medical Diagnosis Chatbot.")
    user_symptoms = set()
    all_asked_symptoms = set()

    # Collect initial symptoms
    while len(user_symptoms) < 5:
        symptom_input = input("Please enter a symptom (or type 'done' if no more): ").strip().lower()
        if symptom_input == 'done' and len(user_symptoms) > 0:
            break
        elif symptom_input != 'done':
            user_symptoms.add(symptom_input)
            all_asked_symptoms.add(symptom_input)

    # Suggest additional symptoms if less than 3 symptoms are provided
    while len(user_symptoms) < 5:
        suggested_symptom = suggest_symptom(user_symptoms, disease_symptom_data, symptom_severity_data, all_asked_symptoms)
        if not suggested_symptom:
            break  # No more symptoms to suggest
        response = input(f"Are you also experiencing {suggested_symptom}? (yes/no): ").strip().lower()
        all_asked_symptoms.add(suggested_symptom)
        if response == "yes":
            print(f"Added {suggested_symptom} to your symptoms.")
            user_symptoms.add(suggested_symptom)

    # Check for diagnoses
    diagnoses = a_star_search(user_symptoms, disease_symptom_data, symptom_severity_data)
    if diagnoses:
        print_diagnoses(diagnoses, disease_description_data, symptom_precaution_data)
        return

    # Continue suggesting symptoms if heuristic threshold not met
    while not diagnoses:
        suggested_symptom = suggest_symptom(user_symptoms, disease_symptom_data, symptom_severity_data, all_asked_symptoms)
        if not suggested_symptom:
            print("Unable to suggest any more symptoms. Please consult a healthcare professional.")
            break
        response = input(f"Are you also experiencing {suggested_symptom}? (yes/no): ").strip().lower()
        all_asked_symptoms.add(suggested_symptom)
        if response == "yes":
            user_symptoms.add(suggested_symptom)
            diagnoses = a_star_search(user_symptoms, disease_symptom_data, symptom_severity_data)
            if diagnoses:
                print_diagnoses(diagnoses, disease_description_data, symptom_precaution_data)
                break

if __name__ == "__main__":
    chatbot(disease_symptom_data, symptom_precaution_data, symptom_severity_data, disease_description_data)
