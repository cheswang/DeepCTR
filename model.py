import tensorflow as tf
from deepctr.models import DCNMix, FwFM

from parameter_server import create_in_process_cluster
from config import parse_args

def get_model(args, linear_feature_columns, dnn_feature_columns):
    gpus = ['GPU:' + str(i) for i in range(args.num_gpus)]
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpus)
    with mirrored_strategy.scope():
        if args.model_type == 'dcnv2':
            model = DCNMix(linear_feature_columns, dnn_feature_columns,task='binary', dnn_hidden_units=args.dnn_layers)
        elif args.model_type == 'fwfm':
            model = FwFM(linear_feature_columns, dnn_feature_columns,task='binary', dnn_hidden_units=args.dnn_layers)
        else:
             print('#### Error! model type must in dcnv2 or fwfm')
             return
        model.summary()
        model.compile("adam", "binary_crossentropy",
                    metrics=['binary_crossentropy'], )
    
    return model
    
def get_ps_model(args, linear_feature_columns, dnn_feature_columns):
    strategy= create_in_process_cluster(args.num_workers, args.num_ps)
    with strategy.scope():
        model = DCNMix(linear_feature_columns, dnn_feature_columns,task='binary', dnn_hidden_units=args.dnn_layers)
        model.summary()
        model.compile("adam", "binary_crossentropy",
                    metrics=['binary_crossentropy'], )
    
    return model
