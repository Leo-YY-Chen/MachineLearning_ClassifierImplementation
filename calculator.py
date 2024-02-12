import numpy as np
import classifier





class Performance_Calculator(classifier.Calculator_Interface):
    def __init__(self):
        self.metrics = classifier.Metrics()

    def calculate_metrics(self, labels, predictions, features=None, weights=None):
        self.metrics.accuracy = self.calculate_accuracy(labels, predictions)
        self.metrics.loss = self.calculate_accuracy(features, weights, labels, predictions)
        return self.metrics






    def calculate_accuracy(self, labels, predictions):
        return np.sum(predictions == labels) / len(labels)
    
    def calculate_loss(self, features, weights, labels, predictions):
        if weights is None: return None
        # By perceptron learning rule, loss function 
        #       L(w) := sum of abs(X*w)
        # where 
        #       X := [features, ones_vector]        2D array (MISCLASSIFIED_feature_instance, n_feature_type + 1)
        #       w := weights                        1D array (n_feature_type + 1)
        X = np.append(features, np.ones([features.shape[0], 1]), axis=1)
        Xw = np.dot(X, weights)
        return np.sum(np.abs(Xw[labels != predictions]))







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
    def test_shuffle_ith_feature_column():
        fic = Feature_Importance_Calculator()
        origin = np.array(np.resize(range(10*7), (10,7)))
        result = fic.shuffle_ith_feature_column(6, origin)
        #print(origin,result)
        if (result[:,0:5] == origin[:,0:5]).all() and (result[:,6] != origin[:,6]).any():
            print("passing")
        else:
            print("fail")
    test_shuffle_ith_feature_column()