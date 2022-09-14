from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

from pathlib import Path

from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1

# ---- Multi-Layer Perceptron script ----
# ---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
# ------------------------------------------------------

# ---- objection initialization section ---------------
# data_obj = Dataset_Loader('toy', '')
# data_obj.dataset_source_folder_path = 'data/stage_1_data/'
# data_obj.dataset_source_file_name = 'toy_data_file.txt'
# My code
# initialize training and test datasets

# Russell Chien: Put your folder path here for testing data
data_folder_path = 'data/stage_3_data/'
data_file_name = 'MNIST'

data_obj = Dataset_Loader('data', '')
data_obj.dataset_source_folder_path = Path(data_folder_path)
data_obj.dataset_source_file_name = data_file_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize CNN Object
if data_file_name == 'MNIST':
    method_obj = Method_CNN('convolution layers', '', 1)
elif data_file_name == 'CIFAR':
    method_obj = Method_CNN('convolution layers', '', 3)
elif data_file_name == 'ORL':
    method_obj = Method_CNN('convolution layers', '', 3)


# Use cuda for model
# Comment out if you don't know what cuda is
# Make sure all lines with ".cuda()" are commented out in Method_MLP.py
# method_obj = method_obj.cuda()

# My code
# Load train dataset and test dataset and create a dictionary with labels 'train' and 'test'.
# 'train' and 'test' are names used in stage_3_code/Method_CNN.py
# Store dictionary into method_obj's (CNN_Method object) data variable
# data has the form: {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
data = data_obj.load()

if data_file_name == 'ORL':
    new_train_y = []
    new_test_y = []
    for element in data['train']['y']:
        new_train_y.append(element - 1)
    for element in data['test']['y']:
        new_test_y.append(element - 1)
    trans_data = {'train': {'X': data['train']['X'], 'y': new_train_y}, 'test': {'X': data['test']['X'], 'y': new_test_y}}
    method_obj.data = trans_data
else:
    method_obj.data = data

method_obj.dataset_name = data_file_name
# Call run function for CNN model object
# CNN model object functions located in stage_3_code/Method_CNN.py
method_obj.to(device)

# Store results of CNN
test_results = method_obj.run()

# Prep results for stat
predictions = test_results['pred_y']
expected = test_results['true_y']

# Output Stat Measurements
accuracy_evaluator = Evaluate_Accuracy('accuracy training evaluator', '')
precision_evaluator = Evaluate_Precision('precision (micro) training evaluator', '')
recall_evaluator = Evaluate_Recall('recall training evaluator', '')
f1_evaluator = Evaluate_F1('f1 (micro) training evaluator', '')

accuracy_evaluator.data = {'true_y': expected, 'pred_y': predictions}
precision_evaluator.data = {'true_y': expected, 'pred_y': predictions}
recall_evaluator.data = {'true_y': expected, 'pred_y': predictions}
f1_evaluator.data = {'true_y': expected, 'pred_y': predictions}

print('Overall Accuracy: ' + str(accuracy_evaluator.evaluate()))
print('Overall Precision: ' + str(precision_evaluator.evaluate()))
print('Overall Recall: ' + str(recall_evaluator.evaluate()))
print('Overall F1: ' + str(f1_evaluator.evaluate()))

# Russell Chien: Put your folder path here for results
result_folder_path = Path('result/stage_3_result/')
result_folder_name = 'CNN_prediction_result'

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = result_folder_path
result_obj.result_destination_file_name = result_folder_name

print('************ Finish ************')
