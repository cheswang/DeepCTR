import time
import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras

class time_callback(keras.callbacks.Callback):
    def __init__(self):
        self.start_time = 0
        self.batch_time = []

    def on_train_batch_begin(self, batch, logs=None):
        self.start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        self.batch_time.append(time.time() - self.start_time)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_q_list(filepathlist, filetype):
    filepathlist = filepathlist
    filepaths = []
    labels = []
    for path in filepathlist:
        data_files = os.listdir(path)
        for data in data_files:
            if data.endswith(filetype):
                data_file = os.path.join(path, data)
                data_label = os.path.basename(os.path.normpath(path))
                filepaths.append(data_file)
                labels.append(data_label)

    return filepaths, labels 

def tables_to_TF(queue_list, tf_filename, file_type='csv'):
    # Target variable needs to be the last column of data
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    all_feature_col = sparse_features + dense_features + ['label']

    def int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    writer = tf.io.TFRecordWriter(tf_filename)
    for file in tqdm(queue_list):
        if file_type == 'csv':
            data = pd.read_csv(file)
        for i in range(len(data)):
            features=tf.train.Features(feature={
                name:int_feature(data[name][i]) for name in all_feature_col})
            example=tf.train.Example(features=features)
            writer.write(example.SerializeToString())
    writer.close()

def pd_to_tfrecord(df):
    filename = './Data/Criteo_virtual_df'+ '.csv'
    df.to_csv(filename)

    filepathlist = ['./Data']
    q, _ = make_q_list(filepathlist, '.csv')
    tffilename = 'Criteo_virtual_TFR.tfrecords'
    tables_to_TF(q, tffilename, file_type='csv')