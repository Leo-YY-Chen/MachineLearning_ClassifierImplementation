import pandas as pd
import numpy as np

class Data_Processor:
    def __init__(self):
        pass
        
    def load(self, filepath):
        return None
    
    def preprocess(self, features, labels):
        return None
    





    def relabel_into_minus_or_plus_one(self, labels):
        return None

    def do_data_balancing(self, features, labels):
        return None
    
    def get_ith_fold_data(self, folds_number, ith, features, labels):
        ith_fold_range = self.get_ith_fold_range(folds_number, ith, labels)
        return features[:, ith_fold_range], labels[ith_fold_range]
        
    def remove_ith_fold_data(self, folds_number, ith, features, labels):
        the_rest_range = self.get_range_without_ith_fold(folds_number, ith, labels)
        return features[:, the_rest_range], labels[the_rest_range]
    





    def get_ith_fold_range(self, number_folds, fold_index, labels):
        fold_size = int(len(labels) / number_folds)
        start_address = fold_index*fold_size
        end_address = len(labels) if fold_index == number_folds-1 else (start_address + fold_size)
        return [i for i in range(start_address, end_address)]
    
    def get_range_without_ith_fold(self, number_folds, i, labels):
        ith_fold_range = self.get_ith_fold_range(number_folds, i, labels)
        return [i for i in range(len(labels)) if i not in ith_fold_range]
    




if __name__ == "__main__":
    #######################
    # TEST getting data
    #######################
    def test_get_ith_fold_data():
        clf = Data_Processor()
        fea = np.array(np.resize(range(2*7), (2,7)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.get_ith_fold_data(3, 0, fea, lab)
        features1, labels1 = clf.get_ith_fold_data(3, 2, fea, lab)
        if (features == np.array([[0,1],[7,8]])).all() and (labels == np.array([0,1])).all():
            if (features1 == np.array([[4,5,6],[11,12,13]])).all() and (labels1 == np.array([4,5,6])).all():
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
        fea = np.array(np.resize(range(2*7), (2,7)))
        lab = np.array(range(7))
        #print(fea, lab)
        features, labels = clf.remove_ith_fold_data(3, 0, fea, lab)
        features1, labels1 = clf.remove_ith_fold_data(3, 2, fea, lab)
        if (features == np.array([[2,3,4,5,6],[9,10,11,12,13]])).all() and (labels == np.array([2,3,4,5,6])).all():
            if (features1 == np.array([[0,1,2,3],[7,8,9,10]])).all() and (labels1 == np.array([0,1,2,3])).all():
                print("passing")
            else:
                print("fail") 
        else:
            print("fail")
    test_remove_ith_fold_data()