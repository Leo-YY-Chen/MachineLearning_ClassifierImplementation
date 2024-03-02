import numpy as np
import classifier




class Calculator:
    def __init__(self) -> None:
        pass

    # Assume that   labels:      1D array (label)

    def calculate_entropy(self, labels):
        # Entropy := - sum_i=1~n (p_i * log(p_i))
        label_list = list(set(labels))
        prob_list = [np.sum(labels==label)/len(labels) for label in label_list]
        entropy_list = [(0 if prob==0 else prob*np.log(prob)) for prob in prob_list]
        return -np.sum(entropy_list)





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
    





    def calculate_entropy(self, labels):
        # Entropy := - sum_i=1~n (p_i * log(p_i))
        label_list = list(set(labels))
        prob_list = [np.sum(labels==label)/len(labels) for label in label_list]
        entropy_list = [(0 if prob==0 else prob*np.log(prob)) for prob in prob_list]
        return -np.sum(entropy_list)
    
    def calculate_information_gains(self, features, labels, parent_entropy):
        IGs = []
        for i, median in enumerate(np.median(features, axis=0)):
            left_labels = labels[features[:,i] > median]
            right_labels = labels[features[:,i] <= median]
            children_entropy = self.calculate_entropy(left_labels) + self.calculate_entropy(right_labels)
            IGs.append(parent_entropy - children_entropy)
        IGs[IGs==0] = -100 # IG = 0 means the feature owing only two kind of values and it should be ignore  
        return IGs







class Feature_Importance_Calculator:
    def __init__(self):
        self.classifier = None

    # Assume that   features:    2D array (feature_type, feature_value)
    #               labels:      1D array (label)

    def calculate_feature_importances(self, classifier, features, labels, number_repetition=10):
        self.set_classifer(classifier)
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        return [self.get_ith_feature_importance(ith, features, labels, number_repetition) for ith in range(features.shape[1])]
    
    
    def set_classifer(self, classifier:classifier.Classifier):
        self.classifier = classifier


    def get_ith_feature_importance(self, ith, features, labels, number_repetition):
        result = 0
        for repeat in range(number_repetition):
                shuffled_features = self.shuffle_ith_feature_column(ith, features)
                predictions = self.classifier.get_predictions(shuffled_features)
                result += self.calculate_accuracy(labels, predictions) 
        return result/number_repetition

    def shuffle_ith_feature_column(self, ith, features):
        features_copy = features.copy()
        while not self._is_shuffled(features, features_copy):
            np.random.shuffle(features_copy[:, ith])
        return features_copy
    
    def calculate_accuracy(self, labels, predictions):
        return np.sum(predictions == labels) / len(labels)


    def _is_shuffled(self, features, shuffled_features):
        return (features != shuffled_features).any()





if __name__ == "__main__":
    import unittest

    class Test_Calculator(unittest.TestCase):

        def setUp(self) -> None:
            self.calculator = Calculator()

        def test_calculate_entropy(self):
            entropy = self.calculator.calculate_entropy(np.array([1,1,-1,-1]))
            self.assertEqual(entropy, np.log(2))

        def tearDown(self):
            del self.calculator




    class Test_Performance_Calculator(unittest.TestCase):

        def setUp(self) -> None:
            self.calculator = Performance_Calculator()

        def test_take_linear_function(self):
            features = np.array([[1,2],[3,4],[5,6],[0,1]])
            weights = np.array([1,-1,3])
            XW = self.calculator.take_linear_function(features, weights)
            self.assertTrue((XW == np.array([2,2,2,2])).all())

        def test_calculate_L1_gradient(self):
            labels = np.array([1,1,1,1])
            features = np.array([[1,-2],[-3,4],[5,-6],[0,1]])
            weights = np.array([-1,1,0])
            gradient = self.calculator.calculate_L1_gradient(labels, features, weights)
            self.assertTrue((gradient == np.array([-9,12,-1])/4).all())

        def test_calculate_L1_loss(self):
            labels = np.array([1,1,1,1])
            features = np.array([[1,-2],[-3,4],[5,-6],[0,1]])
            weights = np.array([-1,1,0])
            loss = self.calculator.calculate_L1_loss(features, weights, labels)
            self.assertEqual(loss, 5.5)
        
        def tearDown(self):
            del self.calculator





    class Test_Feature_Importance_Calculator(unittest.TestCase):

        def setUp(self) -> None:
            self.calculator = Feature_Importance_Calculator()

        def test_shuffle_ith_feature_column(self):
            origin = np.array(np.resize(range(2*7), (2,7)))
            result = self.calculator.shuffle_ith_feature_column(6, origin)
            self.assertTrue((result[:,0:5] == origin[:,0:5]).all())
            self.assertTrue((result[:,6] != origin[:,6]).any())
        
        def tearDown(self):
            del self.calculator

    unittest.main()