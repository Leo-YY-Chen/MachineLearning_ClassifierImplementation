from datetime import datetime
import numpy as np
import os
import pickle
from typing import Callable, Sequence, Iterable, Any
import data_processor ##### NEED REFINE: does classifier need to depend on data_processor?






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
                 timestamp = f"invalid_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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



class Calculator_Interface:
    def calculate_metrics(self, labels:Iterable[Sequence[Any]], predictions:Iterable[Sequence[Any]]) -> Metrics:
        pass

    def calculate_feature_importances(self, get_predictions:Callable[[Iterable[Sequence[Any]]], Iterable[Any]], 
                                        features:Iterable[Sequence[Any]],
                                        labels:Iterable[Any], 
                                        number_repetition:int) -> Iterable[float]:
        pass





class Presenter_Interface: ##### NEED CHECK: can Performance_Presenter match Presenter_Interface by different func args?
    def show_performance(self, information:Information, metrics:Metrics) -> None:
        pass
    
    def save_performance(self, information:Information, metrics:Metrics) -> None:
        pass





class Classifier:
    def __init__(self):
        self.type = "DecisionTree_Clustering_NeuralNetwork_etc"
        self.timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}

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
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Train")) ##### NEED REFINE: init Information() with too many args
        self.save_classifier()
    
    def test(self, features, labels):
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Test")) ##### NEED REFINE: init Information() with too many args
    
    def valid(self, features, labels):
        self.get_performance(features, labels, Information(self.type, self.timestamp, "Valid")) ##### NEED REFINE: init Information() with too many args

    





    def update_parameters(self, features, labels) -> None:
        pass

    def get_predictions(self, features) -> list[float]:
        pass 
    
    def get_performance(self, features, labels, information:Information):
        metrics = self.calculate_metrics(labels, self.get_predictions(features))
        self.show_performance(information, metrics)
        self.save_performance(information, metrics)
        
    

    





    def calculate_metrics(self, labels, predictions):
        Calculator_Interface().calculate_metrics(labels, predictions)
        
    def show_performance(self, information:Information, metrics:Metrics):
        Presenter_Interface().show_performance(information, metrics)
    
    def save_performance(self, information:Information, metrics:Metrics):
        Presenter_Interface().save_performance(information, metrics)
    
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