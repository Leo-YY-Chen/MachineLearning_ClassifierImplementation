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
        labels = self.relabel_into_minus_or_plus_one(labels)
        features = self.do_min_max_normalization(features)
        #self.do_data_balancing()
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
        ith_fold_range = self.get_ith_fold_range(folds_number, ith, labels)
        return features[ith_fold_range, :], labels[ith_fold_range]
        
    def remove_ith_fold_data(self, folds_number, ith, features, labels):
        the_rest_range = self.get_range_without_ith_fold(folds_number, ith, labels)
        return features[the_rest_range, :], labels[the_rest_range]

    def get_ith_fold_range(self, number_folds, fold_index, labels):
        fold_size = int(len(labels) / number_folds)
        start_address = fold_index*fold_size
        end_address = len(labels) if fold_index == number_folds-1 else (start_address + fold_size)
        return [i for i in range(start_address, end_address)]
    
    def get_range_without_ith_fold(self, number_folds, i, labels):
        ith_fold_range = self.get_ith_fold_range(number_folds, i, labels)
        return [i for i in range(len(labels)) if i not in ith_fold_range]
    



    
    def get_data_bigger_than_median(self, features, labels, major_feature_index):
        median = np.median(features, axis=0)[major_feature_index]
        left_indices = features[:, major_feature_index] > median
        return features[left_indices, :], labels[left_indices]
    
    def get_data_not_bigger_than_median(self, features, labels, major_feature_index):
        median = np.median(features, axis=0)[major_feature_index]
        right_indices = features[:, major_feature_index] <= median
        return features[right_indices, :], labels[right_indices]
    




if __name__ == "__main__":
    #######################
    # TEST getting data
    #######################
    def test_get_ith_fold_data():
        clf = Data_Processor()
        fea = np.array(np.resize(range(2*7), (7,2)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.get_ith_fold_data(3, 0, fea, lab)
        features1, labels1 = clf.get_ith_fold_data(3, 2, fea, lab)
        if (features == np.array([[0,1],[2,3]])).all() and (labels == np.array([0,1])).all():
            if (features1 == np.array([[8,9],[10,11],[12,13]])).all() and (labels1 == np.array([4,5,6])).all():
                print("passing")
            else:
                print("fail") 
        else:
            print("fail")
    test_get_ith_fold_data()
    


    def test_get_ith_fold_range():
        clf = Data_Processor()
        labels = np.array([i for i in range(7)])
        rg = clf.get_ith_fold_range(3, 0, labels)
        rg1 = clf.get_ith_fold_range(3, 2, labels)
        if (rg == [0,1]) and (rg1 == [4,5,6]):
            print("passing")
        else:
            print("fail")
    test_get_ith_fold_range()



    def test_remove_ith_fold_data():
        clf = Data_Processor()
        fea = np.array(np.resize(range(2*7), (7,2)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.remove_ith_fold_data(3, 0, fea, lab)
        features1, labels1 = clf.remove_ith_fold_data(3, 2, fea, lab)
        if (features == np.array([[4,5],[6,7],[8,9],[10,11],[12,13]])).all() and (labels == np.array([2,3,4,5,6])).all():
            if (features1 == np.array([[0,1],[2,3],[4,5],[6,7]])).all() and (labels1 == np.array([0,1,2,3])).all():
                print("passing")
            else:
                print("fail") 
        else:
            print("fail")
    test_remove_ith_fold_data()