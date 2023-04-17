#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <set>

using namespace std;

int main() {
	string csv_file = "dataset.csv";
	ifstream file(csv_file);

	if (!file.is_open()) {
		cerr << "Error opening the file: " << csv_file << endl;
		return 1;
	}

	map<string, vector<string>> disease_data;
	set<string> unique_symptoms;

	string line;
	string disease_name;
	vector<string> symptoms;
	int row_counter = 0;

	while (getline(file, line)) {
		istringstream line_stream(line);

		if (row_counter % 10 == 0) {
			getline(line_stream, disease_name, ',');
			symptoms.clear();
		}
		else {
			line_stream.ignore(numeric_limits<streamsize>::max(), ',');
		}

		string symptom;
		while (getline(line_stream, symptom, ',')) {
			if (!symptom.empty()) {
				symptoms.push_back(symptom);
				unique_symptoms.insert(symptom);
			}
			else {
				break;
			}
		}

		if (row_counter % 10 == 9) {
			disease_data[disease_name] = symptoms;
		}

		row_counter++;
	}

	file.close();

	// Print the disease_data map
	for (const auto &disease : disease_data) {
		cout << "Disease: " << disease.first << endl;
		cout << "Symptoms: ";
		for (const auto &symptom : disease.second) {
			cout << symptom << ", ";
		}
		cout << endl << "-----" << endl;
	}

	// Write unique symptoms to the symptoms_questionnaire.txt file
	ofstream symptoms_file("symptoms_questionnaire.txt");
	if (!symptoms_file.is_open()) {
		cerr << "Error opening the symptoms_questionnaire.txt file." << endl;
		return 1;
	}

	for (const string &symptom : unique_symptoms) {
		symptoms_file << symptom << endl;
	}

	symptoms_file.close();

	// Write prolog_disease_evaluation.txt file
	//ofstream prolog_file("prolog_disease_evaluation.txt");
	//if (!prolog_file.is_open()) {
		//cerr << "Error opening the prolog_disease_evaluation.txt file." << endl;
		//return 1;
	//}

	//for (const auto &disease : disease_data) {
	//	prolog_file << "disease(" << disease.first << ") :-" << endl;
	//	for (const auto &symptom : disease.second) {
		//	prolog_file << "    patient(\"" << symptom << "\", yes)," << endl;
	//	}
	//	prolog_file << "    !." << endl << endl;
	//}

	//prolog_file.close();
	system("pause");
	return 0;
}
