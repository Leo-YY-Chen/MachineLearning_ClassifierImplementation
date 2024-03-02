import sys
sys.path.append('..')
import numpy as np
from classifier import Classifier, Calculator_Interface, Presenter_Interface, Data_Processor_Interface



class LinearClassifier(Classifier):
    def __init__(self, calculator:Calculator_Interface, presenter:Presenter_Interface, lr=0.001, epoch = 10, loss_type="L1"):
        super().__init__(calculator, presenter)
        self.information.type = 'LC'
        self.information.loss_type = loss_type
        self.attributes["parameters"] = {'w': None}
        self.attributes["hyper_parameters"] = {'lr': lr, 'epoch':epoch}
        
        
    def k_fold_cross_validation(self, train_features, train_labels, test_features, test_labels, data_processor:Data_Processor_Interface, k = 3):
        for i in range(k):
            self.set_information(fold_quantity=k, fold_number=i, epoch_quantity=self.attributes["hyper_parameters"]['epoch'])
            for j in range(self.information.epoch_quantity):
                self.set_information(epoch_number=j)
                super().train(*data_processor.remove_ith_fold_data(k, i, train_features, train_labels))
                self.valid(*data_processor.get_ith_fold_data(k, i, train_features, train_labels))
                
        self.test(test_features, test_labels)

    def train(self, features, labels):
        self.set_information(state="Train", epoch_quantity=self.attributes["hyper_parameters"]['epoch'])
        for j in range(self.information.epoch_quantity):
            self.set_information(epoch_number=j)
            self.update_parameters(features, labels)
            self.get_performance(features, labels)
            self.save_classifier()



    
    



    
    def update_parameters(self, features, labels):
        self.check_weights(features) 
        gradient = self.calculator.calculate_gradient(labels, features, self.attributes['parameters']['w'], "L1")
        self.attributes['parameters']['w'] = self.attributes['parameters']['w'] - self.attributes['hyper_parameters']['lr'] * gradient
        
    def get_predictions(self, features):
        self.check_weights(features) 
        Xw = self.calculator.take_linear_function(features, self.attributes['parameters']['w'])
        predictions = np.ones(Xw.shape)
        predictions[Xw < 0] = -1
        return predictions

    def check_weights(self, features):
        if self.attributes['parameters']['w'] is None:
            self.attributes['parameters']['w'] = np.random.rand(features.shape[1] + 1)






    def get_performance(self, features, labels):
        self.set_metrics(labels, self.get_predictions(features), features)
        self.show_performance()
        self.save_performance()

    def set_metrics(self, labels, predictions, features):
        self.metrics = self.calculator.calculate_metrics(labels, predictions, features, 
                                                         self.attributes['parameters']['w'], 
                                                         self.information.loss_type)
        self.reset_train_valid_accuracy_loss()
        self.set_train_valid_accuracy_loss()






    def reset_train_valid_accuracy_loss(self):
        if self.is_looping_starting():
            self.metrics.train_accuracy = []
            self.metrics.train_loss = []
            self.metrics.valid_accuracy = []
            self.metrics.valid_loss = []

    def set_train_valid_accuracy_loss(self):
        if self.information.state == "Train":
            self.metrics.train_accuracy.append(self.metrics.accuracy)
            self.metrics.train_loss.append(self.metrics.loss)
        elif self.information.state == "Valid":
            self.metrics.valid_accuracy.append(self.metrics.accuracy)
            self.metrics.valid_loss.append(self.metrics.loss)

    def is_looping_starting(self):
        if self.information.epoch_number == 0:
            if len(self.metrics.train_accuracy) != 0:
                if len(self.metrics.valid_accuracy)!=0:
                    return True
        else:
            return False