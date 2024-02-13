import numpy as np
import classifier





class Performance_Calculator(classifier.Calculator_Interface):
    def __init__(self):
        self.metrics = classifier.Metrics()

    def calculate_metrics(self, labels, predictions, features=None, weights=None, loss_type="L1"):
        self.metrics.accuracy = self.calculate_accuracy(labels, predictions)
        self.metrics.loss = self.calculate_loss(features, weights, labels, loss_type)
        return self.metrics
    






    def calculate_accuracy(self, labels, predictions):
        return np.sum(predictions == labels) / len(labels)
    
    def calculate_loss(self, features, weights, labels, loss_type="L1"):
        if loss_type == "L1":
            return self.calculate_L1_loss(features, weights, labels)
        else:
            print(f"Current loss_type is {loss_type}. You can setup classifier.information.loss_type for other loss function.")
            exit()
        
    def calculate_gradient(self, labels, features, weights, loss_type="L1"):
        if loss_type == "L1":
            return self.calculate_L1_gradient(labels, features, weights)
        else:
            print(f"Current loss_type is {loss_type}. You can setup classifier.information.loss_type for other loss function.")
            exit()





        
        
    def calculate_L1_loss(self, features, weights, labels):
        if weights is None: return None
        # By Mean Absolute Error/ L1 loss function 
        #       L_L1 := (1/n) * sum_i=1~n( |y_i - f(x_i)| )
        # where 
        #       n:          n_feature_instances     int
        #       y_i:        ith label               1D array (1,)
        #       x_i:        [ith feature, 1]^T      2D array (1, n_feature_type + 1)
        #       f(x_i):=    x_i * w                 2D array (1, 1)
        #       w:          [weights, bias]         1D array (n_feature_type + 1,)
        Xw = self.take_linear_function(features, weights)
        return 1/len(labels) * np.sum(np.abs(labels - Xw))
    
    def calculate_L1_gradient(self, labels, features, weights):
        if weights is None: return None
        # By Mean Absolute Error/ L1 loss function
        #       dL_L1/dw := (1/n) * sum_i=1~n(sign_i*x_i)
        # where
        #       n:          n_feature_instances         int
        #       x_i:        [ith feature, 1]^T          2D array (1, n_feature_type + 1)
        #       sign_i:     (f(x_i)-y_i)/|f(x_i)-y_i|   1D array (1,)   
        #       y_i:        ith label                   1D array (1,)
        X = np.append(features, np.ones([features.shape[0], 1]), axis=1)
        XW = self.take_linear_function(features, weights)
        XW_minus_Y = XW - labels
        sign = [(xw_minus_y/np.abs(xw_minus_y) if xw_minus_y != 0 else 0) for xw_minus_y in XW_minus_Y]
        return 1/len(labels) * np.sum( np.expand_dims(sign, axis=1) * X, axis=0)
    




    
    def take_linear_function(self, features, weights):
        # Linear function 
        #       f(x) := x*w+c = [x, 1]*[w, c] = X*W
        # where
        #       X: [features, ones_vector]          2D array (n_feature_instance, n_feature_type + 1)
        #       W: [weights, bias]                  1D array (n_feature_type + 1, )
        #       X*W:  X dot W                       1D array (n_feature_instance, )
        X = np.append(features, np.ones([features.shape[0], 1]), axis=1)
        XW = np.dot(X, weights)
        return XW
    







class Feature_Importance_Calculator():
    def __init__(self, classifier:classifier.Classifier):
        self.classifier = classifier

    # Assume that   features:    2D array (feature_type, feature_value)
    #               labels:      1D array (label)

    def calculate_feature_importances(self, features, labels, number_repetition=10):
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        return [self.get_ith_feature_importance(self.classifier.get_predictions, ith, features, labels, number_repetition) for ith in range(features.shape[1])]
    
    

    def get_ith_feature_importance(self, ith, features, labels, number_repetition):
        result = 0
        for repeat in range(number_repetition):
                shuffled_features = self.shuffle_ith_feature_column(self, ith, features)
                predictions = self.classifier.get_predictions(shuffled_features)
                result += self.calculate_accuracy(labels, predictions) 
        return result/number_repetition

    def shuffle_ith_feature_column(self, ith, features):
        features_copy = features.copy()
        np.random.shuffle(features_copy[:, ith])
        return features_copy
    
    def calculate_accuracy(self, labels, predictions):
        return np.sum(predictions == labels) / len(labels)








if __name__ == "__main__":
    #######################
    # TEST getting feature importance
    #######################
    '''def test_shuffle_ith_feature_column():
        fic = Feature_Importance_Calculator()
        origin = np.array(np.resize(range(10*7), (10,7)))
        result = fic.shuffle_ith_feature_column(6, origin)
        #print(origin,result)
        if (result[:,0:5] == origin[:,0:5]).all() and (result[:,6] != origin[:,6]).any():
            print("passing")
        else:
            print("fail")
    test_shuffle_ith_feature_column()
'''




    #######################
    # TEST calculator loss, gradient
    #######################
    def test_take_linear_function():
        calculator = Performance_Calculator()
        features = np.array([[1,2],[3,4],[5,6],[0,1]])
        weights = np.array([1,-1,3])

        XW = calculator.take_linear_function(features, weights)
        if (XW == np.array([2,2,2,2])).all():
            print("passing")
        else:
            print("fail")
    test_take_linear_function()

    def test_calculate_L1_gradient():
        calculator = Performance_Calculator()
        labels = np.array([1,1,1,1])
        features = np.array([[1,-2],[-3,4],[5,-6],[0,1]])
        weights = np.array([-1,1,0])
        gradient = calculator.calculate_L1_gradient(labels, features, weights)
        if (gradient == np.array([-9,12,-1])/4).all():
            print("passing")
        else:
            print("fail")
    test_calculate_L1_gradient()

    def test_calculate_L1_loss():
        calculator = Performance_Calculator()
        labels = np.array([1,1,1,1])
        features = np.array([[1,-2],[-3,4],[5,-6],[0,1]])
        weights = np.array([-1,1,0])
        loss = calculator.calculate_L1_loss(features, weights, labels)
        if (loss == 5.5):
            print("passing")
        else:
            print("fail")
    test_calculate_L1_loss()