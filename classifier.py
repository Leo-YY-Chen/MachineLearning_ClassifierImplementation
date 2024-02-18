from datetime import datetime
import numpy as np
import os
import pickle
from typing import Callable, Sequence, Iterable, Any, Tuple, Union




class Hyperparameters:
    def __init__(self, 
                 type = "DecisionTree_Clustering_NeuralNetwork_etc",
                 timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                 state = "Train/Test/Valid", 
                 fold_quantity = None, 
                 fold_number = None, 
                 epoch_quantity = None,
                 epoch_number = None,
                 loss_type = None):
        self.type = type
        self.timestamp = timestamp
        self.state = state
        self.fold_quantity = fold_quantity
        self.fold_number = fold_number
        self.epoch_quantity = epoch_quantity
        self.epoch_number = epoch_number
        self.loss_type = loss_type



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
                 timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                 state = "Train/Test/Valid", 
                 fold_quantity = None, 
                 fold_number = None, 
                 epoch_quantity = None,
                 epoch_number = None,
                 loss_type = None):
        self.type = type
        self.timestamp = timestamp
        self.state = state
        self.fold_quantity = fold_quantity
        self.fold_number = fold_number
        self.epoch_quantity = epoch_quantity
        self.epoch_number = epoch_number
        self.loss_type = loss_type







class Data_Processor_Interface:
    def get_ith_fold_data(k:int, 
                          i:int, 
                          features:Iterable[Sequence[Any]],
                          labels:Iterable[Any]) -> Tuple[Iterable[Sequence[Any]], Iterable[Any]]:
        pass

    def remove_ith_fold_data(k:int, 
                             i:int,
                            features:Iterable[Sequence[Any]],
                            labels:Iterable[Any]) -> Tuple[Iterable[Sequence[Any]], Iterable[Any]]:
        pass


class Calculator_Interface:
    def calculate_metrics(labels:Iterable[Any], 
                          predictions:Iterable[Any], 
                          features:Union[Iterable[Sequence[Any]], None], 
                          weights:Union[Iterable[Any], None]) -> Metrics:
        pass

    def calculate_gradient(labels:Iterable[Any],
                          features:Union[Iterable[Sequence[Any]], None], 
                          weights:Union[Iterable[Any], None]) -> Any:
        pass

    def take_linear_function(features:Union[Iterable[Sequence[Any]], None], 
                          weights:Union[Iterable[Any], None]) -> Any:
        pass


class Presenter_Interface:
    def show_performance(information:Information, metrics:Metrics) -> None:
        pass
    
    def save_performance(information:Information, metrics:Metrics) -> None:
        pass







class Classifier:
    def __init__(self, calculator:Calculator_Interface, presenter:Presenter_Interface):
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}
        self.information = Information()
        self.metrics = Metrics()
        self.calculator = calculator
        self.presenter = presenter

    # Assume that   (train_, test_)features:    2D array (feature_instance, feature_type)
    #               (train_, test_)labels:      1D array (label)

    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, data_processor:Data_Processor_Interface, k = 3):
        for i in range(k):
            self.set_information(fold_quantity=k, fold_number=i)
            self.train(*data_processor.remove_ith_fold_data(k, i, train_features, train_labels))
            self.valid(*data_processor.get_ith_fold_data(k, i, train_features, train_labels))
        self.test(test_features, test_labels)

    def train(self, features, labels):
        self.set_information("Train")
        self.update_parameters(features, labels)
        self.get_performance(features, labels)
        self.save_classifier()
    
    def test(self, features, labels):
        self.set_information("Test")
        self.get_performance(features, labels) 
        
    def valid(self, features, labels):
        self.set_information("Valid")
        self.get_performance(features, labels) 
    




    def update_parameters(self, features, labels) -> None:
        pass

    def get_predictions(self, features) -> list[float]:
        pass 

    def get_performance(self, features, labels):
        self.set_metrics(labels, self.get_predictions(features))
        self.show_performance()
        self.save_performance()
        
    



    def set_metrics(self, labels, predictions):
        self.metrics = self.calculator.calculate_metrics(labels, predictions)
        
    def show_performance(self):
        self.presenter.show_performance(self.information, self.metrics)
    
    def save_performance(self):
        self.presenter.save_performance(self.information, self.metrics)
    
    def save_classifier(self):
        with open(self.get_file_name(), 'wb') as fp:
            pickle.dump(self.attributes, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            self.attributes = pickle.load(fp)

    def set_information(self, state = "Train/Test/Valid", 
                        epoch_quantity = None,
                        epoch_number = None,
                        fold_quantity = None, 
                        fold_number = None):
        self.information.state = state if state != "Train/Test/Valid" else self.information.state
        self.information.fold_quantity = self.information.fold_quantity if fold_quantity is None else fold_quantity
        self.information.fold_number = self.information.fold_number if fold_number is None else fold_number
        self.information.epoch_quantity = self.information.epoch_quantity if epoch_quantity is None else epoch_quantity
        self.information.epoch_number = self.information.epoch_number if epoch_number is None else epoch_number

        if state == "Test":
            self.information.fold_quantity = None

    



    def get_file_name(self):
        return os.path.join(os.getcwd(), 'model/checkpoint', self.get_file_infomation()) + '.pkl'
    
    def get_file_infomation(self):
        return f'{self.information.type}_{self.information.timestamp}_Fold{self.information.fold_number}'

    


    




if __name__ == "__main__":
    #######################
    # TEST get_file_name and save & load classifier
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