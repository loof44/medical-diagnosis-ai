def tree_to_code(tree, feature_names, message):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    disease_input = message
    conf, cnf_dis = check_pattern(chk_dis, disease_input)
    if conf == 1:
        if len(cnf_dis) > 1:
            # Handle disambiguation (use the first item for now)
            disease_input = cnf_dis[0]
        else:
            disease_input = cnf_dis[0]
    else:
        return "Invalid symptom. Please try again."

    num_days = 0  # set num_days to 0 by default

    def recurse(node, depth):
        nonlocal num_days
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
                return f"Unable to find '{present_disease}' in the reduced_data index. Skipping this entry."

            symptoms_exp=[]
            for syms in list(symptoms_given):
                if syms == disease_input:
                    continue
                symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)

            output = ""
            output += f"You may have {present_disease[0]}\n"
            output += f"{description_list[present_disease[0]]}\n"

            if present_disease[0] != second_prediction[0]:
                output += f"Alternatively, you may have {second_prediction[0]}\n"
                output += f"{description_list[second_prediction[0]]}\n"

            precution_list=precautionDictionary[present_disease[0]]
            output += "Take the following measures:\n"
            for  i,j in enumerate(precution_list):
                output += f"{i+1}) {j}\n"

            return output

    return recurse(0, 1)



    
getSeverityDict()
getDescription()
getprecautionDict()

#getInfo()

# red_cols = reduced_data.columns
#tree_to_code(trained_classifier,columns,message)
# print("----------------------------------------------------------------------------------------")







def chat(message):
    response = StringIO()  # To capture the output of tree_to_code
    with redirect_stdout(response):  # Temporarily redirect the output to response
        tree_to_code(trained_classifier, columns,message)
    chat_response = response.getvalue()  # Extract the output from response

    return {'answer': chat_response.strip()}  # Remove any extra whitespace
