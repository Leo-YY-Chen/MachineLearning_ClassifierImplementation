from datetime import datetime
import sys
sys.path.append('..')
import utils
import numpy as np
import config as cfg
import os

class Classifier:
    def __init__(self):
        self.hyper_parameters = None
        self.weights = None
        self.iteration_results = {'train_accuracy': [], 'train_loss': [], 'valid_accuracy': [], 'valid_loss': []}
        self.accuracy = np.NaN
        self.loss = np.NaN
        self.features_importance = []
        self.name = None
        self.filename = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def k_fold_cross_validation(self, dataset, k_fold = 3): ##### REFINE: SO UGLY, QUESTION: when to set filename
        temp_data, test_data = dataset.split_in_ratio_for_k_fold(k_fold)
        for k in range(k_fold):
            print(f"Fold {k}:")
            train_data, valid_data = temp_data.get_kth_fold_datasets(k)
            self.train_valid_then_show_and_save_results(train_data, valid_data, filename_tag=f'KFold{k_fold}-{k}')
            self.test_then_show_and_save_results(test_data)
        return None

    def train_valid_then_show_and_save_results(self, train_data, valid_data, filename_tag='temp'):
        self.train_then_show_and_save_results(train_data, filename_tag)
        self.test_then_show_and_save_results(valid_data, tag='Valid')
        return None

    def train_then_show_and_save_results(self, dataset, filename_tag='temp'):
        self.train(dataset)
        self.set_and_save_filename(filename_tag)
        self.show_and_save_results(tag='Train')
        self.save_model()
        return None
    
    def test_then_show_and_save_results(self, dataset, tag='Test'):
        self.test(dataset)
        self.show_and_save_results(tag)
        return None

    def train(self, dataset):
        self.update_weights(dataset)
        self.set_accuracy_and_loss(dataset)
        return None
    
    def update_weights(self, dataset):
        return None

    def set_accuracy_and_loss(self, dataset):
        self.set_accuracy(dataset)
        self.set_loss(dataset)
        return None

    def set_accuracy(self, dataset):
        predicted_labels = self.predict_labels(dataset)
        accuracy = np.sum(predicted_labels == dataset.labels) / dataset.labels.shape[0]
        self.accuracy = accuracy
        return None
    
    def predict_labels(self, dataset):
        # return labels
        return None
    
    def set_loss(self, dataset):
        return None

    def test(self, dataset): 
        self.set_accuracy_and_loss(dataset)
        return None

    def show_and_save_results(self, tag):
        message = self.get_acc_loss_message(tag)
        self.show_results(message)
        self.save_results(message)
        return None
    
    def get_acc_loss_message(self, tag):
        message = f"{tag}: accuracy = {self.accuracy:.3f}, loss = {self.loss:.3f}"
        return message

    def show_results(self, message):
        # show acc, loss ,or plots
        self.show_message(message)
        return None
    
    def save_results(self, message):
        # save acc, loss ,or plots
        self.save_message(message)
        return None
    
    def show_message(self, message):
        print(message)
        return None
    
    def save_message(self, message):
        filepath = os.path.join(cfg.absolute_ML_HW1_path, 'results\log', self.name)
        with open(filepath +'.txt', 'a') as f:
                f.write(f"{message} \n")
        return None
    
    def show_and_save_plots(self):
        # LC: acc list(train, valid, test), loss list(train, valid, test)
        filepath = os.path.join(cfg.absolute_ML_HW1_path, f'results\plots\loss_{self.filename}')
        utils.save_plot(filepath +'.png', self.iteration_results['train_loss'], self.iteration_results['valid_loss'], 'loss')

        filepath = os.path.join(cfg.absolute_ML_HW1_path, 'results\plots', f'acc_{self.filename}')
        utils.save_plot(filepath +'.png', self.iteration_results['train_accuracy'], self.iteration_results['valid_accuracy'], 'accuracy')
        return None

    def set_and_save_filename(self, tag='temp'):
        self.set_filename(tag=tag)
        self.save_filename()
        return None
    
    def set_filename(self, tag='temp'):
        self.filename = f'{self.name}_{tag}_{self.timestamp}'
        return None

    def save_filename(self):
        filepath = os.path.join(cfg.absolute_ML_HW1_path, 'results\log', self.name)
        with open(filepath +'.txt', 'a') as f:
                f.write(f"------------------------------------------------------------------------------- \n")
                f.write(f"{self.filename} \n")
        return None

    def save_model(self):
        dictionary = {'hyper_parameters':self.hyper_parameters, 'weights': self.weights}
        filepath = os.path.join(cfg.absolute_ML_HW1_path, 'model\checkpoint', self.filename)
        utils.save_dict(filepath + '.csv', dictionary)
        return None
  
    def set_show_and_save_feature_importance(self, train_data, number_repetition=10): ##### refine: depend on cfg, too many lines
        self.set_features_importance(train_data, number_repetition)
        filepath = os.path.join(cfg.absolute_ML_HW1_path, f'results\plots', f'feature_importance_{self.filename}')
        utils.save_bar_chart(filepath, cfg.feature_names, self.features_importance) 
        message = self.get_feature_importance_message()
        self.show_message(message)
        self.save_message(message)
        return None
    
    def get_feature_importance_message(self): ##### refine: depend on cfg. and too ugly
        message = f""
        for i in range(len(self.features_importance)):
            message = message + f"{cfg.feature_names[i] + ': ':<25}{self.features_importance[i]:.3f} \n"
        return message

    def set_features_importance(self, dataset, number_repetition=10): ##### refine: too many lines
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        print(f"Start calculating pertubation importance by testing accuracy:")
        self.features_importance = np.zeros(dataset.features.shape[1])
        for feature_index in range(dataset.features.shape[1]):
            for i in range(number_repetition):
                shuffle_dataset_i = dataset
                np.random.seed(i)
                np.random.shuffle(shuffle_dataset_i.features.values[:, feature_index])
                self.features_importance[feature_index] += self.get_testsing_accuracy(shuffle_dataset_i)
        self.features_importance = self.get_testsing_accuracy(dataset) - self.features_importance/number_repetition
        return None
    
    def get_testsing_accuracy(self, dataset):
        self.test(dataset)
        score = self.accuracy
        return score
    

class Classifier_v2:
    def __init__(self):
        self.name = None
        self.hyper_parameters = None
        self.parameters = None

        self.

    # Assume that   k_train_features:   3D array (k, feature_type, feature_value)
    #               k_train_labels:     2D array (k, label)
    #               (test_)features:    2D array (feature_type, feature_value)
    #               (test_)labels:      1D array (label)

    def k_fold_cross_validation(self, k_train_features, k_train_labels, test_features, test_labels, k = 3):
        for i in range(k):
            self.train(self.get_ith_fold_train_features_and_labels(i, k_train_features, k_train_labels))
            self.valid(self.get_ith_fold_valid_features_and_labels(i, k_train_features, k_train_labels))
        self.test(test_features, test_labels)
        return None
    




    def train(self, features, labels):
        self.update_parameters(features, labels)
        self.show_results(features, labels)
        self.save_results(features, labels)
        return None 
    
    def test(self, features, labels):
        self.show_results(features, labels)
        return None
    
    def valid(self, features, labels):
        return self.test(features, labels)




    
    def show_results(self, features, labels):
        self.set_classifier_name()
        self.show_classifier_name()
        # print accuracy and loss
        # plot accuracy and loss if classifier is NN
        return None
    
    def save_results(self, features, labels):        
        # save name, log(plot), and parameters
        return None
    
    def update_parameters(self, features, labels):
        return None

    def calculate_accuracy(self, features, labels):
        return None
    
    def calculate_loss(self, features, labels):
        return None
    




    def get_ith_fold_train_features_and_labels(self, i, k_train_features, k_train_labels):
        # CAUTION: Not Implemented!
        return 
    
    def get_ith_fold_valid_features_and_labels(self, i, k_train_features, k_train_labels):
        # CAUTION: Not Test if output is 2D array!
        return k_train_features[i,:,:], k_train_labels[i,:,:]
    
    def set_classifier_name(self):
        self.name = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return None
    
    def show_classifier_name(self):
        print(f"{self.name}")
        return None



        

    
        
    

if __name__ == "__main__":
    '''dataset = utils.Dataset()
    dataset.load('../data/train.csv')
    dataset.preprocess()
    train_dataset, test_dataset = dataset.split_in_ratio()

    clf = Classifier()
    clf.set_show_and_save_feature_importance(train_dataset)'''

    



    def test_get_ith_fold_train_features_and_labels():
        clf = Classifier()
        a = np.random.rand(2,3,4)
        b = 
        return a == b 

    print(f"{test_get_ith_fold_train_features_and_labels()}:test_get_ith_fold_train_features_and_labels()")