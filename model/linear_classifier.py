import numpy as np
import sys
sys.path.append('..')
import utils
from model.classifier import Classifier


## create linear classifier
class LinearClassifier(Classifier):
    def __init__(self, lr=0.001, epoch = 50):
        super().__init__()
        self.weights = {'w': np.array([None])}
        self.hyper_parameters = {'lr': lr, 'epoch':epoch}
        self.name = 'LC'

    def train_valid_then_show_and_save_results(self, train_dataset, test_dataset, filename_tag='temp'): 
        self.set_and_save_filename(filename_tag)
        for i in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(train_dataset)
            self.valid_in_a_epoch(test_dataset)
            self.show_and_save_results(tag='valid', epoch_id = i)
        self.show_and_save_plots()
        self.save_model()
        return None
    
    def train_in_a_epoch(self, dataset):
        super().train(dataset)
        self.set_iteration_results(tag='train')
        return None
    
    def valid_in_a_epoch(self, dataset):
        self.test(dataset)
        self.set_iteration_results(tag='valid')
        return None

    def train_then_show_and_save_results(self, dataset, filename_tag='temp'): 
        self.set_and_save_filename(filename_tag)
        for i in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(dataset)
            self.show_and_save_results(tag='train_results_only', epoch_id = i)
        self.show_and_save_plots()
        self.save_model()
        return None
    
    def train(self, dataset):
        for j in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(dataset)
        return None

    def update_weights(self, dataset): ##### refine: do more than one thing in a function (i.e. check_or_init_weights)
        self.check_or_init_weights(dataset) 
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        # w(t+1) = w(t) - (sum of delta_x* x^{T}) where delta_x := +1/-1 as x is misclassified as +1/-1
        delta_x = self.get_delta_x(dataset)
        x = self.get_misclassified_x(dataset)
        self.weights['w'] = self.weights['w'] - self.hyper_parameters['lr'] * np.sum(delta_x* x, axis=0)
        return None

    def predict_labels(self, dataset): ##### refine: do more than one thing in a function (i.e. check_or_init_weights)
        self.check_or_init_weights(dataset) 
        Xw = self.apply_linear_function(dataset)
        labels = np.ones(Xw.shape)
        labels[Xw < 0] = -1
        return labels
    
    def check_or_init_weights(self, dataset):
        if self.weights['w'].all() == None:
            self.weights['w'] = np.random.rand(dataset.features.shape[1] + 1)
        return None
   
    def set_loss(self, dataset): 
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        Xw = self.apply_linear_function(dataset)
        indices = self.get_indices_of_the_misclassified(dataset)
        self.loss = np.sum(np.abs(Xw[indices]))
        return None



    def set_iteration_results(self, tag):
        if self.is_iteration_finished():
            self.iteration_results[tag + '_accuracy'] = []
            self.iteration_results[tag + '_loss'] = []
        self.iteration_results[tag + '_accuracy'].append(self.accuracy)
        self.iteration_results[tag + '_loss'].append(self.loss)
        return None
    
    def is_iteration_finished(self):
        if len(self.iteration_results['train_accuracy']) == self.hyper_parameters['epoch'] or len(self.iteration_results['valid_accuracy']) == self.hyper_parameters['epoch']:
            return True
        else:
            return False
        
    def show_and_save_results(self, tag, epoch_id = 0):
        message = self.get_acc_loss_message(tag, epoch_id)
        self.show_results(message)
        self.save_results(message)
        return None
    
    def get_acc_loss_message(self, tag, epoch_id = 0):
        message = f"Epoch {epoch_id}: (Train) accuracy = {self.iteration_results['train_accuracy'][-1]:.3f}, loss = {self.iteration_results['train_loss'][-1]:.3f}"
        if tag=='valid':message += f" (Valid) accuracy = {self.iteration_results['valid_accuracy'][-1]:.3f}, loss = {self.iteration_results['valid_loss'][-1]:.3f}"
        return message
    
    def apply_linear_function(self, dataset):
        # linear function f(x)= x^{T}* w+ c= [x_T, 1]^{T}* [w, c] = X* W
        X = np.append(dataset.features, np.ones([dataset.features.shape[0], 1]), axis=1)
        Xw = np.dot(X, self.weights['w'])
        return Xw

    def get_indices_of_the_misclassified(self, dataset): 
        labels = self.predict_labels(dataset)
        indiecs_misclassifed = (labels != dataset.labels)
        return indiecs_misclassifed

    def get_delta_x(self, dataset):
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        # w(t+1) = w(t) - (sum of delta_x* x^{T}) where delta_x := +1/-1 as x is misclassified as +1/-1
        indices = self.get_indices_of_the_misclassified(dataset)
        delta_x = -1* dataset.labels[indices]
        delta_x = np.expand_dims(delta_x, axis=1)
        return delta_x
    
    def get_misclassified_x(self, dataset):
        indices = self.get_indices_of_the_misclassified(dataset)
        x = np.append(dataset.features, np.ones([dataset.features.shape[0], 1]), axis=1)
        x = x[indices, :]
        return x



if __name__ == '__main__': 
    X = np.random.randn(1000,8)
    Y = np.random.randn(1000,)
    Y[Y>=0] = 1
    Y[Y<0] = -1

    dataset = utils.Dataset(X, Y)
    train_data, test_data = dataset.split_in_ratio()

    lc = LinearClassifier()
    #lc.train_and_show_results(dataset)
    lc.k_fold_cross_validation(dataset)