# Medical Diagnosis Chatbot

A simple Python-based medical diagnosis chatbot that uses heuristic search and a knowledge base of symptoms, precautions, severity, and disease descriptions to help users identify potential diseases based on their symptoms.

## Dependencies

- pandas (https://pandas.pydata.org/)

## Usage

1. Prepare four CSV files with the following content:

   - `dataset.csv`: Contains diseases and their associated symptoms.
   - `symptom_precaution.csv`: Contains diseases and their associated precautions.
   - `symptom_severity.csv`: Contains symptoms and their severity level.
   - `disease_description.csv`: Contains diseases and their descriptions.

2. Install the required dependencies using pip:

pip install pandas


3. Run the chatbot:

python chatbot.py


4. Follow the chatbot's prompts to enter your symptoms and answer questions about additional symptoms. The chatbot will then suggest potential diagnoses, along with their descriptions and precautions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


