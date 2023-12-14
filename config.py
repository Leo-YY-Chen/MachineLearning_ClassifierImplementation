'''
Revise the attributes to run main.py. 
'''

# Data attributes. For simplicity, we only select numerical features for learning. 
datapath = './data/train.csv'
feature_names = ['policy_tenure',
                'age_of_car',
                'age_of_policyholder',
                'population_density',
                'make',
                'airbags',
                'displacement',
                'cylinder',
                'gear_box',
                'turning_radius','length',
                'width', 'height', 'gross_weight', 'ncap_rating']
label_names = ['is_claim']
train_test_split_ratio = [0.001, 0.999]


# Classifier Attributes 
clf_name = "LC" # LC, KNN, DT, PDT
### CAUTION: CHECK THE MODEL & MODIFY ITS ATTRS in main.py. (e.g. n_neighbor in KNNClassifier)
### CAUTION: CHECK THE MODEL & MODIFY ITS ATTRS in main.py. (e.g. n_neighbor in KNNClassifier)
### CAUTION: CHECK THE MODEL & MODIFY ITS ATTRS in main.py. (e.g. n_neighbor in KNNClassifier)




# Cross-validation Attributes
k_fold = "0" # 0, 3, 5, 10