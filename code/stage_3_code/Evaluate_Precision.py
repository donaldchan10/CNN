'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.stage_3_code.evaluate import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None
    
    def evaluate(self):
        print('Evaluating Precision...')
        return precision_score(self.data['true_y'], self.data['pred_y'], average='micro')
