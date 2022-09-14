'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        # dataset has form: {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(loaded_data['train']['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = np.array(loaded_data['train']['X'])[train_index], np.array(loaded_data['train']['X'])[test_index]
            y_train, y_test = np.array(loaded_data['train']['y'])[train_index], np.array(loaded_data['train']['y'])[test_index]
        
            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()
            
            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()
            
            self.acc_evaluate.data = learned_result
            score_list.append(self.acc_evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        