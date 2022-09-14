'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

# Donald Chan : Modified for Stage 3 assignment
import pickle

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        trainX = []
        trainy = []
        testX = []
        testy = []

        file = self.dataset_source_folder_path / self.dataset_source_file_name
        f = open(file, 'rb')
        data = pickle.load(f)
        f.close()

        # trainX = Form = [[image1][image2]...[image n]]
        # trainy = Form = [label1, label2, ..., label n]
        for element in data['train']:
            #trainX = array representation of image
            #trainy = label
            trainX.append(element['image'])
            trainy.append(element['label'])

        
        for element in data['test']:
            #testX = array representation of image
            #testy = label
            testX.append(element['image'])
            testy.append(element['label'])

        return {'train': {'X': trainX, 'y': trainy}, 'test': {'X': testX, 'y': testy}}
