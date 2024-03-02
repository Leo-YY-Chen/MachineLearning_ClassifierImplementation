import pandas as pd
import numpy as np
import classifier






class Data_Processor(classifier.Data_Processor_Interface):
    def __init__(self):
        pass


    def load_csv(self, filepath, feature_names, label_names):
        # Assume that   (train_, test_)features:    2D array (feature_instance, feature_type)
        #               (train_, test_)labels:      1D array (label)
        features = pd.read_csv(filepath, usecols = feature_names).to_numpy()
        labels = pd.read_csv(filepath, usecols = label_names).to_numpy()
        return features, np.squeeze(labels)






    def preprocess(self, features, labels):
        return self.do_data_balancing(self.rescale(features), self.relabel(labels))






    
    def relabel(self, labels):
        labels = self.relabel_into_minus_or_plus_one(labels)
        return labels
    

    def rescale(self, features):
        features = self.do_min_max_normalization(features)
        return features
    
    def do_data_balancing(self, features, labels):
        return features, labels









    def relabel_into_minus_or_plus_one(self, labels):
        labels[labels > 0] = 1
        labels[labels <= 0] = -1
        return labels
    
    def do_min_max_normalization(self, features):
        min = np.min(features, axis=0)
        max = np.max(features, axis=0)
        return (features-min)/(max-min)

     
    
    





    def get_ith_fold_data(self, folds_number, ith, features, labels):
        ith_fold_range = self._get_ith_fold_range(folds_number, ith, labels)
        return features[ith_fold_range, :], labels[ith_fold_range]
        
    def remove_ith_fold_data(self, folds_number, ith, features, labels):
        the_rest_range = self._get_range_without_ith_fold(folds_number, ith, labels)
        return features[the_rest_range, :], labels[the_rest_range]





    def get_data_bigger_than_median(self, features, labels, major_feature_index):
        median = np.median(features, axis=0)[major_feature_index]
        left_indices = features[:, major_feature_index] > median
        return features[left_indices, :], labels[left_indices]
    
    def get_data_not_bigger_than_median(self, features, labels, major_feature_index):
        median = np.median(features, axis=0)[major_feature_index]
        right_indices = features[:, major_feature_index] <= median
        return features[right_indices, :], labels[right_indices]





    def _get_ith_fold_range(self, number_folds, fold_index, labels):
        fold_size = int(len(labels) / number_folds)
        start_address = fold_index*fold_size
        end_address = len(labels) if fold_index == number_folds-1 else (start_address + fold_size)
        return [i for i in range(start_address, end_address)]
    
    def _get_range_without_ith_fold(self, number_folds, i, labels):
        ith_fold_range = self._get_ith_fold_range(number_folds, i, labels)
        return [i for i in range(len(labels)) if i not in ith_fold_range]
    



    
    
    




if __name__ == "__main__":
    import unittest

    class TestData_Processor(unittest.TestCase):

        def setUp(self) -> None:
            self.dp = Data_Processor()
            self.fea = np.array(np.resize(range(2*7), (7,2)))
            self.lab = np.array([0,-1,2,-3,4,-5,6])




        def test_relabel(self):
            labels = self.dp.relabel(self.lab)
            self.assertTrue((labels == np.array([-1,-1,1,-1,1,-1,1])).all())



        
        def test_rescale(self):
            features = self.dp.rescale(self.fea)
            self.assertTrue((features == np.array([[0, 0],
                                                   [2/12, 2/12],
                                                   [4/12, 4/12],
                                                   [6/12, 6/12],
                                                   [8/12, 8/12],
                                                   [10/12,10/12],
                                                   [12/12,12/12]])).all())








        def test_get_ith_fold_data_by_fisrt_fold(self):
            features, labels = self.dp.get_ith_fold_data(3, 0, self.fea, self.lab)
            self.assertTrue((features == np.array([[0,1],[2,3]])).all())
            self.assertTrue((labels == np.array([0,-1])).all())

        def test_get_ith_fold_data_by_last_fold(self):
            features1, labels1 = self.dp.get_ith_fold_data(3, 2, self.fea, self.lab)
            self.assertTrue((features1 == np.array([[8,9],[10,11],[12,13]])).all())
            self.assertTrue((labels1 == np.array([4,-5,6])).all())

        def test_remove_ith_fold_data_by_first_fold(self):
            features, labels = self.dp.remove_ith_fold_data(3, 0, self.fea, self.lab)
            self.assertTrue((features == np.array([[4,5],[6,7],[8,9],[10,11],[12,13]])).all())
            self.assertTrue((labels == np.array([2,-3,4,-5,6])).all())

        def test_remove_ith_fold_data_by_last_fold(self):
            features1, labels1 = self.dp.remove_ith_fold_data(3, 2, self.fea, self.lab)
            self.assertTrue((features1 == np.array([[0,1],[2,3],[4,5],[6,7]])).all())
            self.assertTrue((labels1 == np.array([0,-1,2,-3])).all())

        def tearDown(self):
            del self.dp
            del self.fea
            del self.lab
        
    unittest.main()