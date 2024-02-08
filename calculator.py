import numpy as np

class Metrics:
    def __init__(self):
        self.accuracy = np.nan
        self.confusion_matrix = {'TP':np.nan, 'TN':np.nan, 'FP':np.nan, 'FN':np.nan}
        self.precision = np.nan
        self.recall = np.nan
        self.F1_score = np.nan
        self.ROC = None
        self.cross_entropy = np.nan






class Performance_Calculator:
    def __init__(self):
        self.metrics = Metrics()

    def calculate_metrics(self, labels, predictions):
        return None
    




    def calculate_accuracy(self, labels, predictions):
        self.metrics.accuracy = np.sum(predictions == labels) / len(labels)
        return self.metrics.accuracy






class Feature_Importance_Calculator:
    def __init__(self):
        pass

    # Assume that   features:    2D array (feature_type, feature_value)
    #               labels:      1D array (label)
    #               classifier:  Classifier()


    def calculate_feature_importances(self, trained_classifier, features, labels, number_repetition=10):
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        return [self.get_ith_feature_importance(trained_classifier, ith, features, labels, number_repetition) for ith in range(features.shape[1])]
    
    




    def get_ith_feature_importance(self, trained_classifier, ith, features, labels, number_repetition):
        result = 0
        performance_calculator = Performance_Calculator()
        for repeat in range(number_repetition):
                shuffled_features = self.shuffle_ith_feature_column(self, ith, features)
                predictions = trained_classifier.get_prediction(shuffled_features)
                result += performance_calculator.calculate_accuracy(labels, predictions)
        return result/number_repetition

    def shuffle_ith_feature_column(self, ith, features):
        features_copy = features.copy()
        np.random.shuffle(features_copy[:, ith])
        return features_copy
    




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