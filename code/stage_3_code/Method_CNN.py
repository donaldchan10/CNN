from code.base_class.method import method
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1

class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    batch_size = 1000

    dataset_name = ''

    # CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #From Pytorch Example
    def __init__(self, mName, mDescription, channels):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1).to(self.device)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1).to(self.device)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1).to(self.device)

        self.fc1 = ''
        self.fc2 = ''
        self.fc3 = ''
        self.fc4 = ''
        self.drop = torch.nn.Dropout(p=0.3)

        self.soft = nn.Softmax(dim=1)

    #From Pytorch Example
    def forward(self,x):
        conv1 = self.pool(F.relu(self.conv1(x))).to(self.device)
        conv2 = self.pool(F.relu(self.conv2(conv1))).to(self.device)
        conv3 = self.pool(F.relu(self.conv3(conv2))).to(self.device)
        conv4 = self.pool(F.relu(self.conv4(conv3))).to(self.device)
        flat = torch.flatten(conv4, 1) # flatten dimensions
        activation1 = self.drop(F.relu(self.fc1(flat)).to(self.device))
        activation2 = self.drop(F.relu(self.fc2(activation1)).to(self.device))
        activation3 = self.drop(F.relu(self.fc3(activation2)).to(self.device))
        activation4 = self.fc4(activation3).to(self.device)
        y_pred = self.soft(activation4)
        return y_pred

    def train(self, X, y):
        # X has form: [[image1][image2]...[image n]]
        # y has form: [label1, label2, ..., label n]

        #Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(self.max_epoch + 1):  # you can do an early stop if self.max_epoch is too much...
            #Convert X and y to tensors so pytorch can operate on it

            if self.dataset_name == 'CIFAR' or self.dataset_name == 'ORL':
                #CIFAR or ORL dataset
                tensorX = torch.FloatTensor(np.array(X)).to(self.device)
            else:
                #MNIST dataset
                tensorX = torch.FloatTensor(np.array(X)).unsqueeze(3).to(self.device)

            tensorY = torch.LongTensor(np.array(y)).to(self.device)


            permutation = torch.randperm(tensorX.size()[0]).to(self.device)

            pred = torch.empty(0).to(self.device)
            true = torch.empty(0).to(self.device)

            for i in range(0, tensorX.size()[0], self.batch_size):
                indicies = permutation[i:i+self.batch_size]
                miniX, miniy = tensorX[indicies], tensorY[indicies]

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()

                temp = torch.permute(miniX, (0, 3, 1, 2)).to(self.device)
                y_pred = self.forward(temp).to(self.device)
                # convert y to torch.tensor as well
                y_true = miniy

                #y_pred = y_pred - 1
                #print(y_pred[0])
                #y_true = y_true - 1


                # calculate the training loss
                train_loss = loss_function(y_pred, y_true).to(self.device)

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                # Keep track of pred and true y values to calulate accuracy later
                pred = torch.cat((pred, y_pred), 0)
                true = torch.cat((true, y_true), 0)

            # Create evaluation objects that represent evaluation metrics
            accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
            precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
            recall_evaluator = Evaluate_Recall('recall training evaluator', '')
            f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

            if epoch % 10 == 0:
                accuracy_evaluator.data = {'true_y': true.to('cpu'), 'pred_y': pred.to('cpu').max(1)[1]}
                precision_evaluator.data = {'true_y': true.to('cpu'), 'pred_y': pred.to('cpu').max(1)[1]}
                recall_evaluator.data = {'true_y': true.to('cpu'), 'pred_y': pred.to('cpu').max(1)[1]}
                f1_evaluator.data ={'true_y': true.to('cpu'), 'pred_y': pred.to('cpu').max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(),
                      'Precision: ', precision_evaluator.evaluate(), 'Recall: ', recall_evaluator.evaluate(),
                      'F1 (Micro): ', f1_evaluator.evaluate())


    def test(self, X):
        # do the testing, and result the result
        if self.dataset_name == 'CIFAR' or self.dataset_name == 'ORL':
            #CIFAR
            tensorX = torch.FloatTensor(np.array(X)).to(self.device)
        else:
            tensorX = torch.FloatTensor(np.array(X)).unsqueeze(3).to(self.device)

        temp = torch.permute(tensorX, (0, 3, 1, 2)).to(self.device)
        y_pred = self.forward(temp).to(self.device)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        #data has form:
        # {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        print('method running...')
        if self.dataset_name == 'MNIST':
            self.batch_size = 1000

            self.fc1 = nn.Linear(256, 100).to(self.device)
            self.fc2 = nn.Linear(100, 50).to(self.device)
            self.fc3 = nn.Linear(50, 25).to(self.device)
            self.fc4 = nn.Linear(25, 10).to(self.device)
        elif self.dataset_name == 'CIFAR':
            self.batch_size = 1000

            self.fc1 = nn.Linear(1024, 500).to(self.device)
            self.fc2 = nn.Linear(500, 300).to(self.device)
            self.fc3 = nn.Linear(300, 100).to(self.device)
            self.fc4 = nn.Linear(100, 10).to(self.device)
        elif self.dataset_name == 'ORL':
            self.batch_size = 360
            self.max_epoch = 500

            self.fc1 = nn.Linear(8960, 1000).to(self.device)
            self.fc2 = nn.Linear(1000, 500).to(self.device)
            self.fc3 = nn.Linear(500, 100).to(self.device)
            self.fc4 = nn.Linear(100, 40).to(self.device)
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        pred_y = pred_y.to('cpu')
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}