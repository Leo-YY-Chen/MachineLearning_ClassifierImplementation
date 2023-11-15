import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
import sys
sys.path.append('..')
import function as fn
from tqdm import tqdm

class KNNClassifier():
    def __init__(self, n_neighbor=3, metric='minkowski', p=2):
        # param for model construction
        self.metric = metric
        self.p = p
        self.n_neighbor = n_neighbor

        # param for coding convinience
        self.X_train = []
        self.Y_train = []
        self.model_name = 'KNN'
        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')

    def fit(self, X, Y):
        # Fit the model by input training data (X, Y)
        self.X_train = X
        self.Y_train = Y
        return None

    def distance(self, x_test):
        # Calculate Minkwoski distance between (x_test, X_train) by power (self.p) 
        dist = np.linalg.norm(self.X_train-x_test, ord=self.p, axis=1)     
        return dist

    def pred(self, X_test, Y_test):
        # Make a prediction (self.output) by the model and input testing data (X,Y)
        X_pred = []
        for i in tqdm(range(X_test.shape[0])):
            x = X_test[i,:]
            x_dist = self.distance(x)
            x_neighbor_id = np.argsort(x_dist)[:self.n_neighbor]
            x_pred = 1 if np.sum(self.Y_train[x_neighbor_id]) > 0 else -1
            X_pred.append(x_pred)

        # Calculate the result (acc)
        acc = np.sum(X_pred == Y_test) / Y_test.shape[0]
        return acc

    def train(self, X_train, X_valid, Y_train, Y_valid, k_fold=1, k_fold_iter=0):
        # Given (epoch) number, train and valid the model by (X_train, X_valid, Y_train, Y_valid)
        self.fit(X_train, Y_train)
        acc = self.pred(X_valid, Y_valid)
        print(f"valid_acc = {acc}")

        # Save the acc
        tag = f'kfold{k_fold}-{k_fold_iter}'
        filepath = f'{self.model_name}_{tag}_{self.time}'
        lists = {"acc":acc}
        fn.save_dict('../results/acc_loss_lists/' + filepath + '.csv', lists)
        
        # Save the model
        param = {"metric":self.metric, "p":self.p, "n_neighbor":self.n_neighbor}
        fn.save_dict('./checkpoint/' + filepath + '.csv', param)

    def train_k_fold(self, X, Y, k_fold = 3):
        kf = KFold(n_splits=k_fold, random_state=83, shuffle=True)
        for i, (train_index, valid_index) in enumerate(kf.split(X,y=Y)):
            print(f"Fold {i}:")
            self.train( X[train_index], X[valid_index], Y[train_index], Y[valid_index], k_fold=k_fold, k_fold_iter=i)


if __name__ == '__main__': 
    X = np.random.randn(6,8)
    Y = np.random.randn(6,)
    Y[Y>=0] = 1
    Y[Y<0] = -1
    #print(f"X = {X}")
    #print(f"Y = {Y}")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=83)
    #print(x_test.shape)

    cls = KNNClassifier()
    cls.train(x_train, x_test, y_train, y_test)

    cls2 = KNNClassifier()
    cls2.train_k_fold(X, Y)