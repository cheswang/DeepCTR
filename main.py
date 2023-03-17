import logging
import os
import time

# from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
from deepctr.models import DCNMix

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from datareader import read_data
from config import parse_args
from data_preprocess import preprocess, model_input_config
from model import get_model, get_ps_model
from virtual_data_generator import gen_data_df, gen_data_dataset
from utils import time_callback

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
    args = parse_args()
    data = gen_data_df(args)
    print(data)

    dnn_feature_columns, linear_feature_columns, feature_names = model_input_config(args, data)

    if args.use_ps != 0:
        model = get_ps_model(args, dnn_feature_columns=dnn_feature_columns, linear_feature_columns=linear_feature_columns)
    else:
        model = get_model(args, dnn_feature_columns=dnn_feature_columns, linear_feature_columns=linear_feature_columns)
    print(model)
    # return

    train_model_input = {name:data[name].values for name in feature_names}
    target = ['label']
    perf = time_callback()

    model.fit(train_model_input, data[target].values,
                batch_size=args.batch_size * args.num_gpus, epochs=args.epochs, callbacks=[perf], validation_split=0)

    perf_time = perf.batch_time
    if len(perf_time) < 2:
        print('###### Error! batch * epoch < 2, cant warmup')
        return
    throughout = args.batch_size * args.num_gpus / (sum(perf_time[2: ]) / len(perf_time[2:]))
    print('###### throughout =%.2f (examples/s)'%(throughout))

if __name__ == '__main__':
    main()
