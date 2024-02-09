from datetime import datetime
import numpy as np
import os
import pickle
import data_processor
import calculator
import presenter






class Metrics:
    def __init__(self):
        self.accuracy = np.nan
        self.confusion_matrix = {'TP':np.nan, 'TN':np.nan, 'FP':np.nan, 'FN':np.nan}
        self.precision = np.nan
        self.recall = np.nan
        self.F1_score = np.nan
        self.ROC = None
        self.cross_entropy = np.nan
        self.loss = np.nan

        self.train_accuracy = []
        self.train_loss = []
        self.valid_accuracy = []
        self.valid_loss = []






class Information:
    def __init__(self, 
                 type = "DecisionTree_Clustering_NeuralNetwork_etc",
                 timestamp = None,
                 state = "Train/Test/Valid", 
                 fold_quantity = np.nan, 
                 fold_number = np.nan, 
                 epoch_quantity = np.nan,
                 epoch_number = np.nan):
        self.type = type
        self.timestamp = timestamp
        self.state = state
        self.fold_quantity = fold_quantity
        self.fold_number = fold_number
        self.epoch_quantity = epoch_quantity
        self.epoch_number = epoch_number






class Classifier:
    def __init__(self):
        self.type = "DecisionTree_Clustering_NeuralNetwork_etc"
        self.timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}
        self.metrics = Metrics()

    # Assume that   (train_, test_)features:    2D array (feature_type, feature_value)
    #               (train_, test_)labels:      1D array (label)

    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, k = 3):
        dpc = data_processor.Data_Processor()
        for i in range(k):
            self.train(dpc.get_ith_fold_data(k, i, train_features, train_labels))
            self.valid(dpc.remove_ith_fold_data(k, i, train_features, train_labels))
        self.test(test_features, test_labels)

    def train(self, features, labels):
        self.update_parameters(features, labels)
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Train"))
        self.save_classifier()
    
    def test(self, features, labels):
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Test"))
    
    def valid(self, features, labels):
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Valid"))

    





    def update_parameters(self, features, labels):
        pass

    def get_prediction(self, features):
        pass #return labels
    
    def get_performance(self, features, labels, information:Information):
        self.set_metrics(self.calcualte_performance_metrics(labels, self.get_prediction(features)))
        self.show_performance(information, self.metrics)
        self.save_performance(information, self.metrics)
        
    

    





    def calcualte_performance_metrics(self, labels, prediction):
        performance_calculator = calculator.Performance_Calculator()
        return performance_calculator.calculate_metrics(labels, prediction)
    
    def set_metrics(self, metrics:Metrics):
        self.metrics = metrics
        
    def show_performance(self, information:Information, metrics:Metrics):
        prstr = presenter.Performance_Presenter(information, metrics)
        prstr.show_result()
        pass
    
    def save_performance(self, information:Information, metrics:Metrics):
        prstr = presenter.Performance_Presenter(information, metrics)
        prstr.save_result()
        pass
    
    def save_classifier(self):
        with open(self.get_file_name(), 'wb') as fp:
            pickle.dump(self.attributes, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            self.attributes = pickle.load(fp)


    

    

    def get_file_name(self):
        return os.path.join(os.getcwd(), 'model/checkpoint', self.timestamp) + '.pkl'
    
    
    







        

    
        
    

if __name__ == "__main__":
    #######################
    # TEST save and load classifier
    #######################
    '''def test_save_and_load_classifier():
        clf = Classifier()
        clf.attributes = {'hyper_parameters':{'k': 10, 'lr': 0.001}, 'parameters':{'weight':list(range(7)) ,'bias':np.array(range(7))}}
        filename = clf.get_file_name()

        clf.save_classifier()
        clf.attributes = {'hyper_parameters':{}, 'parameters':{}}
        if len(clf.attributes['hyper_parameters']) == 0 and len(clf.attributes['parameters']) == 0:
            clf.load_classifier(filename)
            if (clf.attributes['hyper_parameters'] == {'k': 10, 'lr': 0.001}) and (clf.attributes['parameters']['weight'] == list(range(7))) and (clf.attributes['parameters']['bias']==np.array(range(7))).all():
                print("passing")
            else:
                print("loading fail")
        else:
            print("saving fail")
    test_save_and_load_classifier()'''