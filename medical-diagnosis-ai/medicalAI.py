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
graph.render("decision_tree")  # This will save the decision tree visualization as a PDF file
Image(graph.pipe(format='png'))  # This will display the decision tree visualization in Jupyter Notebooks



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








# def tree_to_code(tree, feature_names, user_symptom):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]

#     chk_dis = ",".join(feature_names).split(",")
#     symptoms_present = []

#     disease_input = user_symptom

#     def recurse(node, depth):
#         indent = "  " * depth
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]

#             if name == disease_input:
#                 val = 1
#             else:
#                 val = 0
#             if val <= threshold:
#                 recurse(tree_.children_left[node], depth + 1)
#             else:
#                 symptoms_present.append(name)
#                 recurse(tree_.children_right[node], depth + 1)
#         else:
#             present_disease = print_disease(tree_.value[node])

#             red_cols = reduced_data.columns

#             try:
#                 symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
#             except KeyError:
#                 return [f"Unable to find '{present_disease}' in the reduced_data index. Skipping this entry."]

#             result = ["Are you experiencing any of the following symptoms?"]

#             for syms in list(symptoms_given):
#                 result.append(syms)

#             return result

#     return recurse(0, 1)


# def main(user_input):
#     global severityDictionary, description_list, precautionDictionary
#     getSeverityDict()
#     getDescription()
#     getprecautionDict()

#     response = ""
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     # Replace the `getInfo()` function with a user_input variable
#     # getInfo()
#     user_symptom = user_input.strip()

#     try:
#         result = tree_to_code(trained_classifier, columns, user_symptom)
#         response = '\n'.join(result)
#     except Exception as e:
#         response = f"An error occurred: {str(e)}"

#     return response














def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("How many days have you had this symptom? : "))
            break
        except:
            print("Enter valid input.")
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
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            #symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            try:
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            except KeyError:
                print(f"Unable to find '{present_disease}' in the reduced_data index. Skipping this entry.")
                return

            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)

    
# getSeverityDict()
# getDescription()
# getprecautionDict()
# getInfo()

red_cols = reduced_data.columns
# tree_to_code(trained_classifier,columns)
# print("----------------------------------------------------------------------------------------")



def make_prediction(symptom_input, num_days):
    results = {}
    
    def recurse(node, depth):
        nonlocal results
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptom_input:
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

            try:
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            except KeyError:
                results["error"] = f"Unable to find '{present_disease}' in the reduced_data index. Skipping this entry."
                return results

            symptoms_exp = []
            for syms in list(symptoms_given):
                if syms in symptom_input:
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)

            calc_condition(symptoms_exp, num_days)

            if present_disease[0] == second_prediction[0]:
                results["disease"] = present_disease[0]
                results["description"] = description_list[present_disease[0]]
            else:
                results["disease"] = f"{present_disease[0]} or {second_prediction[0]}"
                results["description"] = f"{description_list[present_disease[0]]}\n{description_list[second_prediction[0]]}"

            precaution_list = precautionDictionary[present_disease[0]]
            results["precautions"] = precaution_list

    tree_ = dt_classifier.tree_
    feature_name = [
        columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    symptoms_present = []
    recurse(0, 1)

    return results


