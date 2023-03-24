import os
import portpicker
import time
import multiprocessing
import json

import tensorflow as tf
import tensorflow.keras as keras

from parameter_server import create_in_process_cluster
from config import parse_args

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定该代码文件的可见GPU为第一个和第二个
import numpy as np
print(tf.__version__)#查看tf版本
gpus=tf.config.list_physical_devices('GPU')
print(gpus)#查看有多少个可用的GPU

def create_in_process_cluster(num_workers, num_ps, job_name):
  """Creates and starts local servers and returns the cluster_resolver."""
  # worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  # ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  # cluster_dict = {}
  # cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  # if num_ps > 0:
  #   cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]


  # cluster_dict_json = json.dumps(cluster_dict)
  # with open('ps_port.json', 'w') as f:
  #   f.write(cluster_dict_json)
  with open('./ps_port.json', 'r') as f:
    cluster_dict = json.loads(f.read())
  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  # worker_config = tf.compat.v1.ConfigProto()
  # if multiprocessing.cpu_count() < num_workers + 1:
  #   worker_config.inter_op_parallelism_threads = num_workers + 1

  # if job_name =='worker':
  #   for i in range(num_workers):
  #     tf.distribute.Server(
  #         cluster_spec,
  #         job_name="worker",
  #         task_index=i,
  #         config=worker_config,
  #         protocol="grpc")

  # elif job_name == 'ps':
  #   for i in range(num_ps):
  #     tf.distribute.Server(
  #         cluster_spec,
  #         job_name="ps",
  #         task_index=i,
  #         protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

def create_node(num_workers, num_ps, job_name):
  with open('./ps_port.json', 'r') as f:
    cluster_dict = json.loads(f.read())
  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  if job_name =='worker':
    for i in range(num_workers):
      tf.distribute.Server(
          cluster_spec,
          job_name="worker",
          task_index=i,
          config=worker_config,
          protocol="grpc")

  elif job_name == 'ps':
    for i in range(num_ps):
      tf.distribute.Server(
          cluster_spec,
          job_name="ps",
          task_index=i,
          protocol="grpc")

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

args = parse_args()
NUM_WORKERS = 3
NUM_PS = 2
if args.job_name in ['ps', 'worker']:
  cluster_resolver = create_node(NUM_WORKERS, NUM_PS, args.job_name)
else:
  cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS, args.job_name)
  variable_partitioner = (
      tf.distribute.experimental.partitioners.MinSizePartitioner(
          min_shard_bytes=(256 << 10),
          max_shards=NUM_PS))

  strategy = tf.distribute.ParameterServerStrategy(
      cluster_resolver,
      variable_partitioner=variable_partitioner)

  global_batch_size = 64

  x = tf.random.uniform((10, 10))
  y = tf.random.uniform((10,))

  dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
  dataset = dataset.batch(global_batch_size)
  dataset = dataset.prefetch(2)

  with strategy.scope():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

    model.compile(tf.keras.optimizers.legacy.SGD(), loss="mse", steps_per_execution=10)

  model.fit(dataset, epochs=5, steps_per_epoch=20)
