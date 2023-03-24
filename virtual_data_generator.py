
import numpy as np
import pandas as pd
import tensorflow as tf
    
from config import parse_args

def gen_data_dims(args):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]
    cardinaltity = args.sparse_feature_cardinaltity
    sparse_dims = {sparse_features[i]: cardinaltity[i] for i in range(26)}
    
    return sparse_dims, dense_features
        
    
def create_virtual_data(args):
    feature_dims, dense_feature = gen_data_dims(args)
    total_data_size = args.num_batch * args.batch_size

    sparse_feature = {name: np.random.randint(0, dims, size=(1, total_data_size), dtype=np.int32).squeeze() for name, dims in feature_dims.items()}
    dense_feature = {name: np.random.randint(0, 1000, size=(1, total_data_size), dtype=np.int32).squeeze() for name in dense_feature}
    label = {'label': np.random.randint(0, 2, size=(1, total_data_size), dtype=np.int8).squeeze()}
    
    return sparse_feature, dense_feature, label


def gen_data_df(args):
    sparse_feature, dense_feature, label = create_virtual_data(args)
    sparse_data = pd.DataFrame(sparse_feature)
    dense_data = pd.DataFrame(dense_feature)
    label = pd.DataFrame(label)
    data = pd.concat([sparse_data, dense_data, label], axis=1)
    return data

def gen_data_dataset(args, data, target):
    target = data.pop(target[0])
    dataset = tf.data.Dataset.from_tensor_slices((data.to_dict('list'), target.values))
    return dataset

if __name__ == '__main__':
    args = parse_args()
    data = gen_data_df(args)