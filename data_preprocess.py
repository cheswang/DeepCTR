import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


def preprocess(data):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0,)

    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    return data

def model_input_config(args, data):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]
    # if args.use_hash == 1:
    #     fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=int(args.embedding_hash_size), use_hash=True, dtype='int32', embedding_dim=args.embedding_dims)
    #                     for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
    #                     for feat in dense_features]
    # else:
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max()+1, embedding_dim=args.embedding_dims)
                    for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                    for feat in dense_features]
    
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    return dnn_feature_columns, linear_feature_columns, feature_names