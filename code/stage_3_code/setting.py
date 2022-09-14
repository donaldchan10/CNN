'''
Base SettingModule class for all experiment settings
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc


# -----------------------------------------------------
class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''

    setting_name = None
    setting_description = None

    dataset = None
    method = None
    result = None
    acc_evaluate = None
    pre_evaluate = None
    recall_evaluate = None
    f1_evaluate = None

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription

    def prepare(self, sDataset, sMethod, sResult, aEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.acc_evaluate = aEvaluate


    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', accuracy evaluation:',
              self.acc_evaluate.evaluate_name)

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return
