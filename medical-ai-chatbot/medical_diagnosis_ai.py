import pandas as pd


# Load the CSV files
dataset = pd.read_csv("dataset.csv", header=None)
symptom_precaution = pd.read_csv("symptom_precaution.csv", header=None)
symptom_severity = pd.read_csv("symptom_severity.csv", header=None)
disease_description = pd.read_csv("disease_description.csv", header=None)

# Preprocess dataset.csv data
disease_symptom_data = {}
# Iterate over rows and store symptoms in a set for each disease
for index, row in dataset.iterrows():
    disease = row[0]
    symptoms = [symptom.strip() for symptom in row[1:].dropna().tolist()]

    if disease not in disease_symptom_data:
        disease_symptom_data[disease] = set(symptoms)
    else:
        disease_symptom_data[disease].update(symptoms)

# Create dictionaries from the data
symptom_precaution_data = {}
# Store precautions for each disease
for index, row in symptom_precaution.iterrows():
    disease = row[0]
    precautions = row[1:].dropna().tolist()
    symptom_precaution_data[disease] = precautions
# Create dictionaries for symptom severity and disease description
symptom_severity_data = dict(symptom_severity.values.tolist())
disease_description_data = dict(disease_description.values.tolist())
# Heuristic function to calculate the similarity between user and disease symptom
def heuristic(user_symptoms, disease_symptoms):
    common_symptoms = user_symptoms.intersection(disease_symptoms)
    return len(common_symptoms) / len(user_symptoms)
# A* search algorithm to find potential diagnoses
def a_star_search(user_symptoms, disease_data, symptom_severity_data, threshold=0.8):
    potential_diagnoses = []
    # Iterate over diseases and calculate heuristic value
    for disease, symptoms in disease_data.items():
        disease_symptoms = set(symptoms)
        h = heuristic(user_symptoms, disease_symptoms)
        # If the heuristic value is above the threshold, add to potential diagnoses
        if h >= threshold:
            max_severity = max([symptom_severity_data.get(symptom, 0) for symptom in disease_symptoms])
            potential_diagnoses.append((disease, h, max_severity))
    # Sort potential diagnoses based on heuristic value and severity
    potential_diagnoses.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    return potential_diagnoses
# Function to suggest a new symptom for the user
def suggest_symptom(user_symptoms, disease_data, symptom_severity_data, answered_symptoms):
    symptom_counts = {}
    # Count the occurrences of each remaining symptom
    for _, symptoms in disease_data.items():
        common_symptoms = user_symptoms.intersection(symptoms)
        if common_symptoms:
            remaining_symptoms = symptoms.difference(user_symptoms).difference(answered_symptoms)
            for symptom in remaining_symptoms:
                if symptom in symptom_severity_data:
                    if symptom not in symptom_counts:
                        symptom_counts[symptom] = 0
                    symptom_counts[symptom] += len(common_symptoms)
    # Sort symptoms based on count and severity
    sorted_symptoms = sorted(symptom_counts.items(), key=lambda x: (-x[1], symptom_severity_data.get(x[0], 0)), reverse=True)

    return sorted_symptoms[0][0] if sorted_symptoms else None


MIN_SYMPTOMS_COUNT = 1
# Main chatbot function
def chatbot(disease_symptom_data, symptom_precaution_data, symptom_severity_data, disease_description_data):
    print("Welcome to the Medical Diagnosis Chatbot.")
    print("What symptoms are you experiencing? (Please enter them separated by commas): ")
    user_symptoms = set(input("").split(','))
    answered_symptoms = set()

    while True:
        # Suggest a new symptom for the user
        suggested_symptom = suggest_symptom(user_symptoms, disease_symptom_data, symptom_severity_data, answered_symptoms)
        if not suggested_symptom:
            print("Unable to suggest any more symptoms. Please consult a healthcare professional.")
            break
        # Ask user if they are experiencing the suggested symptom
        user_input = input(f"Are you experiencing {suggested_symptom}? (yes/no): ").lower()
        
        if user_input == "yes":
            user_symptoms.add(suggested_symptom)
        elif user_input == "no":
            answered_symptoms.add(suggested_symptom)
        else:
            print("Please respond with 'yes' or 'no'.")
            continue

        answered_symptoms.add(suggested_symptom)
        # Check if enough symptoms are provided for a potential diagnosis
        if len(user_symptoms) >= MIN_SYMPTOMS_COUNT:
            diagnoses = a_star_search(user_symptoms, disease_symptom_data, symptom_severity_data)

            if diagnoses:
                print("\nPotential diagnoses:")
                for diagnosis, similarity, severity in diagnoses:
                    print(f"{diagnosis} (similarity: {similarity * 100:.2f}%, severity: {severity})")
                    print(f"Description: {disease_description_data[diagnosis]}")
                    print("Precautions:")
                    for precaution in symptom_precaution_data[diagnosis]:
                        print(f"- {precaution}")
                    print()
                break
            else:
                print("No matching diseases found yet. Please provide more symptoms.")


if __name__ == "__main__":
    chatbot(disease_symptom_data, symptom_precaution_data, symptom_severity_data, disease_description_data)
