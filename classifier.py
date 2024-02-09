from datetime import datetime
import numpy as np
import os
import pickle
import data_processor
import calculator
import presenter

class Classification_Metrics:
    def __init__(self):
        self.accuracy = np.nan
        self.confusion_matrix = {'TP':np.nan, 'TN':np.nan, 'FP':np.nan, 'FN':np.nan}
        self.precision = np.nan
        self.recall = np.nan
        self.F1_score = np.nan
        self.ROC = None
        self.cross_entropy = np.nan
        self.loss = np.nan

class Classifier_Infomation:
    def __init__(self):
        self.timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.state = "Train/Test/Valid"
        self.fold_quantity = np.nan
        self.fold_number = np.nan
        self.epoch_quantity = np.nan
        self.epoch_number = np.nan

class Classifier:
    def __init__(self):
        self.attributes = {'hyper_parameters':{}, 'parameters':{}}
        self.metrics = Classification_Metrics()
        self.information = Classifier_Infomation()

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
        self.get_performance(features, labels)
        self.save_classifier()
    
    def test(self, features, labels):
        self.get_performance(features, labels)
    
    def valid(self, features, labels):
        self.get_performance(features, labels)

    





    def update_parameters(self, features, labels) -> None:
        pass

    def get_prediction(self, features) -> float:
        pass #return labels
    
    def get_performance(self, features, labels):
        self.metrics = self.calcualte_performance_metrics(labels, self.get_prediction(features))
        self.show_performance()
        self.save_performance()
        
    

    





    def calcualte_performance_metrics(self, labels, prediction):
        performance_calculator = calculator.Performance_Calculator()
        return performance_calculator.calculate_metrics(labels, prediction)
        
    def show_performance(self):
        #prstr = presenter.Result_Presenter(self.information, self.metrics)
        #prstr.show_result()
        pass
    
    def save_performance(self):
        #prstr = presenter.Result_Presenter()
        #prstr.save_result()
        pass
    
    def save_classifier(self):
        with open(self.get_file_name(), 'wb') as fp:
            pickle.dump(self.attributes, fp)
    
    def load_classifier(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as fp:
            self.attributes = pickle.load(fp)


    

    def get_state_message(self, state = "Train/Test/Valid", fold = np.nan, epoch = np.nan):
        fold_info = "" if fold == np.nan else f" Fold {fold}" 
        epoch_info = "" if epoch == np.nan else f" Epoch {epoch}"
        info = state + fold_info + epoch_info
        return f"{info}: accuracy = {self.metrics.accuracy:.3f}, loss = {self.metrics.loss:.3f}"

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


    


    #######################
    # TEST perfomance
    #######################
    def test_get_state_message():
        clf = Classifier()
        message = clf.get_state_message("Train", 1, "number")
        if message == "Train Fold 1 Epoch number: accuracy = nan, loss = nan":
            print("passing")
        else:
            print("fail")
    test_get_state_message()