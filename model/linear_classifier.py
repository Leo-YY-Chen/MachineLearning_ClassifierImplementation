import numpy as np
import sys
sys.path.append('..')
import utils
import classifier as clf


## create linear classifier
class LinearClassifier(clf.Classifier):
    def __init__(self, num_features, lr=0.001, epoch = 50):
        super().__init__()
        self.weights = {'w': np.random.rand(num_features+1)}
        self.hyper_parameters = {'lr': lr, 'epoch':epoch}
        self.name = 'LC'

    def train_valid_then_show_and_save_results(self, train_data, test_data, filename_tag='temp'): 
        self.set_and_save_filename(filename_tag)
        for i in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(train_data)
            self.valid_in_a_epoch(test_data)
            self.show_and_save_results()
        self.show_and_save_plots()
        self.save_model()
        return None
    
    def train_in_a_epoch(self, train_data):
        super().train(train_data)
        self.set_iteration_results(tag='train')
        return None
    
    def valid_in_a_epoch(self, valid_data):
        self.test(valid_data)
        self.set_iteration_results(tag='valid')
        return None

    def train_then_show_and_save_results(self, data, filename_tag='temp'): 
        self.set_and_save_filename(filename_tag)
        for i in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(data)
            self.show_and_save_results(tag='show_train_results_only')
        self.show_and_save_plots()
        self.save_model()
        return None
    
    def train(self, train_data):
        for j in range(self.hyper_parameters['epoch']):
            self.train_in_a_epoch(train_data)
        return None

    def update_weights(self, data): 
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        # w(t+1) = w(t) - (sum of delta_x* x^{T}) where delta_x := +1/-1 as x is misclassified as +1/-1
        delta_x = self.get_delta_x(data)
        x = self.get_misclassified_x(data)
        self.weights['w'] = self.weights['w'] - self.hyper_parameters['lr'] * np.sum(delta_x* x, axis=0)
        return None

    def predict_labels(self, data):
        Xw = self.apply_linear_function(data)
        labels = np.ones(Xw.shape)
        labels[Xw < 0] = -1
        return labels
   
    def set_loss(self, data): 
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        Xw = self.apply_linear_function(data)
        indices = self.get_indices_of_the_misclassified(data)
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
        if len(self.iteration_results['valid_accuracy']) == self.hyper_parameters['epoch']:
            return True
        else:
            return False
    
    def get_acc_loss_message(self, epoch_id = 0, tag='valid'):
        message = f"Epoch {epoch_id}: (Train) accuracy = {self.iteration_results['train_accuracy'][-1]:.3f}, loss = {self.iteration_results['train_loss'][-1]:.3f}"
        if tag =='valid': message += f" (Valid) accuracy = {self.iteration_results['valid_accuracy'][-1]:.3f}, loss = {self.iteration_results['valid_loss'][-1]:.3f}"
        return message
    
    def apply_linear_function(self, data):
        # linear function f(x)= x^{T}* w+ c= [x_T, 1]^{T}* [w, c] = X* W
        X = np.append(data.features, np.ones([data.features.shape[0], 1]), axis=1)
        Xw = np.dot(X, self.weights['w'])
        return Xw

    def get_indices_of_the_misclassified(self, data): 
        labels = self.predict_labels(data)
        indiecs_misclassifed = (labels != data.labels)
        return indiecs_misclassifed

    def get_delta_x(self, data):
        # By perceptron learning rule, loss function L(w) := sum of abs(x^{T}* w) where x is misclassified.
        # w(t+1) = w(t) - (sum of delta_x* x^{T}) where delta_x := +1/-1 as x is misclassified as +1/-1
        indices = self.get_indices_of_the_misclassified(data)
        delta_x = -1* data.labels[indices]
        delta_x = np.expand_dims(delta_x, axis=1)
        return delta_x
    
    def get_misclassified_x(self, data):
        indices = self.get_indices_of_the_misclassified(data)
        x = np.append(data.features, np.ones([data.features.shape[0], 1]), axis=1)
        x = x[indices, :]
        return x



if __name__ == '__main__': 
    X = np.random.randn(1000,8)
    Y = np.random.randn(1000,)
    Y[Y>=0] = 1
    Y[Y<0] = -1

    dataset = utils.Dataset(X, Y)
    train_data, test_data = dataset.split_in_ratio()

    lc = LinearClassifier(X.shape[1])
    #lc.train_and_show_results(dataset)
    lc.k_fold_cross_validation(dataset)