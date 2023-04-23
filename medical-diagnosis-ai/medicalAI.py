import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import graphviz
from IPython.display import Image
import pickle
from io import StringIO
from contextlib import redirect_stdout


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Loading training and testing datasets from CSV files
train_data = pd.read_csv("Data/Training.csv")
test_data = pd.read_csv("Data/Testing.csv")
columns = train_data.columns[:-1]

# Preparing data for training 
X = train_data[columns]
y = train_data['prognosis']

# Reducing data for visualization purposes
reduced_data = train_data.groupby(train_data['prognosis']).max()

# Encoding labels
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y)
y = label_encoder.transform(y)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_test_data = test_data[columns]
y_test_data = label_encoder.transform(test_data['prognosis'])

# Training Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
trained_classifier = dt_classifier.fit(X_train, y_train)

# Cross-validation scores
scores = cross_val_score(trained_classifier, X_test, y_test, cv=3)
print(scores.mean())

# Training Support Vector Machine Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
print("For SVM: ")
print(svm_model.score(X_test, y_test))

# Feature importances
importances = trained_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# Decision tree visualization
dot_data = export_graphviz(trained_classifier, out_file=None, feature_names=columns, class_names=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  
Image(graph.pipe(format='png'))  



# Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)




warnings.filterwarnings("ignore", category=DeprecationWarning)


#--------Function Definitions-------------------#


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(X):
       symptoms_dict[symptom] = index
def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("Please Consult Your Doctor.")
    else:
        print("Symptoms do not indicate any life threatening medical issues, However, precaution is advised.")


def getDescription():
    global description_list
    with open('Data/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)





def getSeverityDict():
    global severityDictionary
    with open('Data/Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Data/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("\nYour Name? \t\t\t\t",end="->")
    name=input("")
    print("Hello, ",name)

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = label_encoder.inverse_transform(val[0])

    return list(map(lambda x:x.strip(),list(disease)))



def tree_to_code(tree, feature_names, message):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    disease_input = message.strip()
    conf, cnf_dis = check_pattern(chk_dis, disease_input)
    if conf == 1:
        disease_input = cnf_dis[0]
    else:
        return {'answer': "Enter valid symptom.", 'action': 'error'}

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns

            try:
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            except KeyError:
                return {'answer': f"Unable to find '{present_disease}' in the reduced_data index. Skipping this entry.", 'action': 'error'}

            symptoms_exp = []
            for syms in list(symptoms_given):
                if syms == disease_input:
                    continue
                symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)

            output = ""
            output += f"You may have {present_disease[0]}\n"
            output += f"{description_list[present_disease[0]]}\n"

            if present_disease[0] != second_prediction[0]:
                output += f"Alternatively, you may have {second_prediction[0]}\n"
                output += f"{description_list[second_prediction[0]]}\n"

            precution_list = precautionDictionary[present_disease[0]]
            output += "Take the following measures:\n"
            for i, j in enumerate(precution_list):
                output += f"{i + 1}) {j}\n"

            return {'answer': output, 'action': 'diagnosis'}

    return recurse(0, 1)






    
getSeverityDict()
getDescription()
getprecautionDict()


#getInfo()

# red_cols = reduced_data.columns
#tree_to_code(trained_classifier,columns)
# print("----------------------------------------------------------------------------------------")







def chat(message, num_days=0):
    if message.lower() == "hi" or message.lower() == "hello":
        response = "Hello! I am an AI-based medical assistant. I can help you identify your illness based on your symptoms. Please tell me the symptom you are experiencing."

    elif message.lower() == "quit":
        response = "Goodbye! If you need further assistance, don't hesitate to ask."

    else:
        response = tree_to_code(trained_classifier, columns, message)

        if not response or response.get('action') == 'error':
            response = {
                'answer': "I'm sorry, I couldn't find a matching disease in the dataset. Please try again with a different symptom or consult a medical professional.",
                'action': 'error'
            }
        elif num_days is not None:
            condition_response = calc_condition(response.get('symptoms_exp', []), num_days)
            response['answer'] += f"\n{condition_response}"

    print("Response:", response)  # Keep this line to print the response before returning it
    return response
