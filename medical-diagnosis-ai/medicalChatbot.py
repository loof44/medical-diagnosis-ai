import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import preprocessing

# Load the trained model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load training data
train_data = pd.read_csv('Data/Training.csv')
columns = train_data.columns[:-1]
X = train_data[columns]

# Symptoms dictionary
symptoms_dict = {}
for index, symptom in enumerate(X):
    symptoms_dict[symptom] = index

# Label encoder
y = train_data['prognosis']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)

def chatbot(user_symptom):
    def tree_to_code(tree, feature_names, user_symptom):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        # Replace the input() with user_symptom
        disease_input = user_symptom.strip()

        result = []

        symptoms_present = []

        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                if name == disease_input:
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                result.extend(present_disease)

        recurse(0, 1)
        return result

    def print_disease(node):
        node = node[0]
        val  = node.nonzero() 
        disease = label_encoder.inverse_transform(val[0])
        return list(map(lambda x:x.strip(),list(disease)))

    return tree_to_code(model, columns, user_symptom)

