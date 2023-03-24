import os
import portpicker
import time
import multiprocessing
import json

import tensorflow as tf
import tensorflow.keras as keras
from DeepCTR.deepctr.models import DCNMix, FwFM
from DeepCTR.deepctr.models.dcnmix import DCNMix_class

from config import parse_args
from virtual_data_generator import gen_data_df, gen_data_dataset
from data_preprocess import preprocess, model_input_config
import numpy as np
print(tf.__version__)#查看tf版本
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定该代码文件的可见GPU为第一个和第二个
gpus=tf.config.list_physical_devices('GPU')
print(gpus)#查看有多少个可用的GPU

def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)

variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

global_batch_size = 64

args = parse_args()
data = gen_data_df(args)
# print(data)
target = data.pop('label')
dataset =  tf.data.Dataset.from_tensor_slices((data.values, target.values))
dataset = dataset.batch(global_batch_size)
dataset = dataset.prefetch(2).repeat()

# for x, y in dataset.take(5):
#   print(x, y)
def get_model():
    model2 = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
    return model2

dnn_feature_columns, linear_feature_columns, feature_names = model_input_config(args, data)
with strategy.scope():
    model2 = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    model2 = get_model()
    model2.compile(tf.keras.optimizers.legacy.SGD(), loss="mse", steps_per_execution=10)

    model = DCNMix_class(linear_feature_columns, dnn_feature_columns,task='binary', dnn_hidden_units=args.dnn_layers)
    # model.summary()
    model.compile("adam", "binary_crossentropy",
                metrics=['binary_crossentropy'], )
    
working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

model2.fit(dataset, epochs=5, steps_per_epoch=10)