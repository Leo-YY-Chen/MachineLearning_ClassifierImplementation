import numpy as np
import classifier 

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

    def caculate_metrics(self, labels, predictions):
        return None
    




    def caculate_accuracy(self, labels, predictions):
        self.metrics.accuracy = np.sum(predictions == labels) / len(labels)
        return self.metrics.accuracy






class Feature_Importance_Calculator:
    def __init__(self):
        self.feature_importances = None

    # Assume that   features:    2D array (feature_type, feature_value)
    #               labels:      1D array (label)
    #               classifier:  Classifier()


    def caculate_feature_importances(self, trained_classifier, features, labels, number_repetition=10):
        # ref: https://scikit-learn.org/stable/modules/permutation_importance.html
        return [self.get_feature_importance_by_index(trained_classifier, index, features, labels, number_repetition) for index in features.shape[1]]
    
    




    def get_feature_importance_by_index(self, trained_classifier, index, features, labels, number_repetition):
        result = 0
        for i in range(number_repetition):
                shuffled_features = self.get_feature_column_shuffled_by_index(self, index, features)
                result[index] += trained_classifier.compute_accuracy(labels, self.get_prediction(shuffled_features))
        return result/number_repetition

    def get_feature_column_shuffled_by_index(self, index, features):
        features_copy = features.copy()
        np.random.shuffle(features_copy[:, index])
        return features_copy
    




if __name__ == "__main__":
    #######################
    # TEST getting feature importance
    #######################
    def test_get_feature_column_shuffled_by_index():
        clf = classifier.Classifier_v2()
        origin = np.array(np.resize(range(10*7), (10,7)))
        result = clf.get_feature_column_shuffled_by_index(6, origin)
        #print(origin,result)
        if (result[:,0:5] == origin[:,0:5]).all() and (result[:,6] != origin[:,6]).any():
            print("passing")
        else:
            print("fail")
    test_get_feature_column_shuffled_by_index()