import os
import json
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.ops.parsing_ops import FixedLenFeature
from DeepCTR.deepctr.estimator import FwFMEstimator, DCNMixEstimator
from DeepCTR.deepctr.estimator.inputs import input_fn_tfrecord
from virtual_data_generator import gen_data_df
from utils import time_callback, pd_to_tfrecord
from config import parse_args


if __name__ == "__main__":

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'chief': ['localhost:34567'],
            'worker': ["localhost:12345", "localhost:12346"],
            'ps': ["localhost:23456"]
        },
        'task': {'type': 'chief', 'index': 0}
    })
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    worker_config = tf.ConfigProto()
    worker_config.inter_op_parallelism_threads = 4

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5' # 指定该代码文件的可见GPU为第一个和第二个
    gpus=tf.config.list_physical_devices('GPU')
    print('*'*20, 'worker:', gpus)#查看有多少个可用的GPU
    for i in range(2):
        tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name="worker",
            task_index=i,
            config=worker_config)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '' # 指定该代码文件的可见GPU为第一个和第二个
    gpus=tf.config.list_physical_devices('GPU')
    print('*'*20, 'ps: ', gpus)#查看有多少个可用的GPU
    for i in range(1):
        tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name="ps",
            task_index=i)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, 1000), 128))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, 1000))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 2.generate input data for model
    feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in dense_features})
    feature_description['label'] = FixedLenFeature(dtype=tf.int64, shape=1)

    args = parse_args()
    data = gen_data_df(args)
    pd_to_tfrecord(data)
    print(data)
    train_model_input = input_fn_tfrecord('./Criteo_virtual_TFR.tfrecords', feature_description, 'label', batch_size=256,
                                          num_epochs=100, shuffle_factor=10)
    test_model_input = input_fn_tfrecord('./Criteo_virtual_TFR.tfrecords', feature_description, 'label',
                                         batch_size=8, num_epochs=1, shuffle_factor=0)
    
    # 3.Define Model,train,predict and evaluate
    strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)
    model = DCNMixEstimator(linear_feature_columns, dnn_feature_columns, task='binary',
                            config=tf.estimator.RunConfig(tf_random_seed=2021, train_distribute=strategy))

    tf.estimator.train_and_evaluate(
        model,
        train_spec=tf.estimator.TrainSpec(input_fn=train_model_input),
        eval_spec=tf.estimator.EvalSpec(input_fn=test_model_input)
    )
    eval_result = model.evaluate(test_model_input)
    print(eval_result)