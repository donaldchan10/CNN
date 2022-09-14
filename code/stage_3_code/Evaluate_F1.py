'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None
    
    def evaluate(self):
        print('Evaluating F1...')
        # Method set to micro
        return f1_score(self.data['true_y'], self.data['pred_y'], average='micro')
