from data_processor import Data_Processor
from classifier_factory import Classifier_Factory
from calculator import Feature_Importance_Calculator

import shap



filepath = './data/train.csv'
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





processor = Data_Processor()
features, labels = processor.load_csv(filepath, feature_names, label_names)
features, labels = processor.preprocess(features, labels)






factory = Classifier_Factory()
hyperparameters = {'lr': 0.00001, 'epoch':5, 'loss_type':"L1"}
classifier0 = factory.make_classifier("LinearClassifier")
classifier1 = factory.make_classifier("LinearClassifier", **hyperparameters)




classifier1.train(features, labels)




FI_calculator = Feature_Importance_Calculator()
print(f"handcraft FI {FI_calculator.calculate_feature_importances(classifier1, features, labels)}")
explainer = shap.Explainer(classifier1.get_predictions, features)
print(f"SHAP FI {explainer.shap_values(features)}")