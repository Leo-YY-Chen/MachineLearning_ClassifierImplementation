from datetime import datetime
import numpy as np
import os
import pickle
from typing import Callable, Sequence, Iterable, Any
import data_processor 






class Metrics:
    def __init__(self):
        self.accuracy = None
        self.confusion_matrix = {'TP':None, 'TN':None, 'FP':None, 'FN':None}
        self.precision = None
        self.recall = None
        self.F1_score = None
        self.ROC = None
        self.cross_entropy = None
        self.loss = None

        self.train_accuracy = []
        self.train_loss = []
        self.valid_accuracy = []
        self.valid_loss = []






class Information:
    def __init__(self, 
                 type = "DecisionTree_Clustering_NeuralNetwork_etc",
                 timestamp = f"invalid_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                 state = "Train/Test/Valid", 
                 fold_quantity = None, 
                 fold_number = None, 
                 epoch_quantity = None,
                 epoch_number = None):
        self.type = type
        self.timestamp = timestamp
        self.state = state
        self.fold_quantity = fold_quantity
        self.fold_number = fold_number
        self.epoch_quantity = epoch_quantity
        self.epoch_number = epoch_number

    def set_information(self, state = "Train/Test/Valid", 
                epoch_quantity = None,
                epoch_number = None,
                fold_quantity = None, 
                fold_number = None, 
                ):
        self.state = state
        self.fold_quantity = fold_quantity
        self.fold_number = fold_number
        self.epoch_quantity = epoch_quantity
        self.epoch_number = epoch_number





class Data_Processor_Interface:
    def get_ith_fold_data(self, 
                          k:int, i:int, 
                          features:Iterable[Sequence[Any]],
                          labels:Iterable[Any]) -> Iterable[Sequence[Any]]: ##### PROBLEM: Check return type
        pass

    def remove_ith_fold_data(self, 
                          k:int, i:int,
                          features:Iterable[Sequence[Any]],
                          labels:Iterable[Any]) -> Iterable[Sequence[Any]]: ##### PROBLEM: Check return type
        pass






class Calculator_Interface:
    def calculate_metrics(self, labels:Iterable[Sequence[Any]], predictions:Iterable[Sequence[Any]]) -> Metrics:
        pass

    def calculate_feature_importances(self, get_predictions:Callable[[Iterable[Sequence[Any]]], Iterable[Any]], 
                                        features:Iterable[Sequence[Any]],
                                        labels:Iterable[Any], 
                                        number_repetition:int) -> Iterable[float]:
        pass





class Presenter_Interface:
    def show_performance(self, information:Information, metrics:Metrics) -> None:
        pass
    
    def save_performance(self, information:Information, metrics:Metrics) -> None:
        pass





class Classifier:
    def __init__(self):
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}
        self.information = Information()
        self.metrics = Metrics()

    # Assume that   (train_, test_)features:    2D array (feature_type, feature_value)
    #               (train_, test_)labels:      1D array (label)

    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, k = 3):
        for i in range(k):
            self.train(Data_Processor_Interface().get_ith_fold_data(k, i, train_features, train_labels)) ##### Problem: how to decide the implementation of interface?
            self.valid(Data_Processor_Interface().remove_ith_fold_data(k, i, train_features, train_labels))
        self.test(test_features, test_labels)

    def train(self, features, labels):
        self.information.set_information("Train")
        self.update_parameters(features, labels)
        self.get_performance(features, labels)
        self.save_classifier()
    
    def test(self, features, labels):
        self.information.set_information("Test")
        self.get_performance(features, labels) 
        
    def valid(self, features, labels):
        self.information.set_information("Valid")
        self.get_performance(features, labels) 
    





    def update_parameters(self, features, labels) -> None:
        pass

    def get_predictions(self, features) -> list[float]:
        pass 

    def get_performance(self, features, labels):
        self.metrics = self.calculate_metrics(labels, self.get_predictions(features))
        self.show_performance()
        self.save_performance()
        
    

    





    def calculate_metrics(self, labels, predictions):
        Calculator_Interface().calculate_metrics(labels, predictions)
        
    def show_performance(self):
        Presenter_Interface().show_performance(self.information, self.metrics)
    
    def save_performance(self):
        Presenter_Interface().save_performance(self.information, self.metrics)
    
    def save_classifier(self):
        with open(self.get_file_name(), 'wb') as fp:
            pickle.dump(self.attributes, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            self.attributes = pickle.load(fp)


    

    

    def get_file_name(self):
        return os.path.join(os.getcwd(), 'model/checkpoint', self.information.timestamp) + '.pkl'
    
    
    








        

    
        
    

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