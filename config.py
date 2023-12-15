import os

# Data attributes. For simplicity, we only select numerical features for learning. 
absolute_ML_HW1_path = os.getcwd()
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

if __name__ == '__main__':
    print(absolute_ML_HW1_path)