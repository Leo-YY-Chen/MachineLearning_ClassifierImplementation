import numpy as np
from sklearn.model_selection import train_test_split, KFold
from datetime import datetime
import sys
sys.path.append('..')
import function as fn


## create linear classifier
class LinearClassifier():
    def __init__(self, num_features, lr=0.001):
        # param for model construction
        self.w = np.random.randn(num_features+1)
        self.lr = lr

        # param for coding convinience
        self.X = []
        self.output = 0
        self.model_name = 'LC'
        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')

    def pred(self, X, Y):
        # Make a prediction (self.output) by the model and input (X,Y)  
        self.X = np.append(X, np.ones([X.shape[0], 1]), axis=1)
        Xw = np.dot(self.X, self.w)
        self.output = np.ones(Xw.shape)
        self.output[Xw < 0] = -1

        # Calculate the results (acc, loss)
        # where loss = Summation_over_mispred_x(delta_x * x^T * w) and delta_x:= 1 if (label_x is mispred as 1) else -1
        Y = np.squeeze(Y)
        acc = np.sum(self.output == Y) / Y.shape[0]
        loss = np.dot(self.output[self.output != Y], Xw[self.output != Y])
        return acc, loss

    def update_weight(self, X, Y):
        # Make a prediction (self.output) by the model and input (X,Y)
        self.pred(X, Y)
        # Update the model weights
        # ref: p9, p10 in https://dmi.unibas.ch/fileadmin/user_upload/dmi/Studium/Computer_Science/Vorlesung_HS19/Pattern_Recongnition/06_Linear_Classifier_1_2.pdf
        self.w = self.w - self.lr * np.sum(self.X[self.output != Y, :], axis=0)
        return None
    
    def fit(self, X, Y, epoch = 50):
        for j in range(epoch):
            self.update_weight(X,Y)           
        return None

    def train(self, X_train, X_valid, Y_train, Y_valid, epoch = 50, k_fold=1, k_fold_iter=0):
        # Given (epoch) number, train and valid the model by (X_train, X_valid, Y_train, Y_valid)
        train_acc_list, train_loss_list = [], []
        valid_acc_list, valid_loss_list = [], []
        min_valid_loss = 9999999999999999999
        for j in range(epoch):
                # train, valid, and show results (acc, loss)
                print(f"Epoch:{j+1}/{epoch}")
                train_acc, train_loss = self.update_weight(X_train,Y_train)
                valid_acc, valid_loss = self.pred(X_valid,Y_valid)
                print(f" train_acc = {train_acc}, train_loss = {train_loss}, valid_acc = {valid_acc}, valid_loss = {valid_loss}, lr = {self.lr}")
                
                # Params for saving results (acc, loss) and setting model checkpoint by (w, lr)
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
                valid_acc_list.append(valid_acc)
                valid_loss_list.append(valid_loss)                
                tag = f'epoch{epoch}_kfold{k_fold}-{k_fold_iter}'
                filepath = f'{self.model_name}_{tag}_{self.time}'
                
                # Save the model checkpoint (epoch, w, lr)
                if valid_loss <= min_valid_loss:
                    min_valid_loss = valid_loss
                    param = {"epoch": j+1, "w": self.w, "lr": self.lr}
                    fn.save_dict('./checkpoint/' + filepath + '.csv', param)

                # Save the plots (acc, loss)
                if j == epoch-1:
                    lists = {"train_acc_list":train_acc_list,"train_loss_list":train_loss_list, "valid_acc_list":valid_acc_list, "valid_loss_list":valid_loss_list}
                    fn.save_dict('../results/acc_loss_lists/' + filepath + '.csv', lists)
                    fn.save_plot('../results/plots/loss_'+ filepath +'.png', train_loss_list, valid_loss_list, 'loss')
                    fn.save_plot('../results/plots/acc_'+ filepath +'.png', train_acc_list, valid_acc_list, 'accuracy')

    def train_k_fold(self, X, Y, epoch = 50, k_fold = 3):
        kf = KFold(n_splits=k_fold, random_state=83, shuffle=True)
        for i, (train_index, valid_index) in enumerate(kf.split(X,y=Y)):
            print(f"Fold {i}:")
            self.train( X[train_index], X[valid_index], Y[train_index], Y[valid_index], epoch = epoch, k_fold=k_fold, k_fold_iter=i)
               


if __name__ == '__main__': 
    X = np.random.randn(6,8)
    Y = np.random.randn(6,)
    Y[Y>=0] = 1
    Y[Y<0] = -1
    print(f"X = {X}")
    print(f"Y = {Y}")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=83)
    
    cls = LinearClassifier()
    cls.train(x_train, x_test, y_train, y_test, epoch=1000)

    cls2 = LinearClassifier()
    cls2.train_k_fold(X, Y, epoch = 50, k_fold = 3)