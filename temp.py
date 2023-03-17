import tensorflow as tf
import tensorflow.keras as keras
import os
import portpicker
import time

from parameter_server import create_in_process_cluster
from config import parse_args


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 指定该代码文件的可见GPU为第一个和第二个
import numpy as np
print(tf.__version__)#查看tf版本
gpus=tf.config.list_physical_devices('GPU')
print(gpus)#查看有多少个可用的GPU

def main(args):
    # fashion_mnist = tf.keras.datasets.fashion_mnist

    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # # 向数组添加维度 -> 新的维度 == (28, 28, 1)
    # # 我们这样做是因为我们模型中的第一层是卷积层
    # # 而且它需要一个四维的输入 (批大小, 高, 宽, 通道).
    # # 批大小维度稍后将添加。
    # train_images = train_images[..., None]
    # test_images = test_images[..., None]

    # # 获取[0,1]范围内的图像。
    # train_images = train_images / np.float32(255)
    # test_images = test_images / np.float32(255)

    # dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels))

    # def input_fn(X,y,shuffle, batch_size):
    #     dataset = tf.data.Dataset.from_tensor_slices((X,y))
    #     if shuffle: 
    #         dataset = dataset.shuffle(buffer_size=100000)
    #     dataset = dataset.repeat()
    #     dataset = dataset.batch(batch_size)
    #     return dataset

    # dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    # dataset = dataset.map(lambda x: x * x) # 1.0, 4.0, 9.0, 16.0, 25.0

    # dataset=input_fn(train_images,train_labels,True, 32)
    # test_dataset=input_fn(test_images,test_labels,True, 32)

    worker_ports = [portpicker.pick_unused_port() for _ in range(1)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(1)]
    chief_ports = [portpicker.pick_unused_port() for _ in range(1)]

    cluster_dict = {}
    cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]
    cluster_dict["chief"] = ["localhost:%s" % port for port in chief_ports]

    import json
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": cluster_dict,
    "task": {"type": args.job_name, "index": 0} #定义本进程为worker节点，即["127.0.0.1:5001"]为计算节点
    })

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    print('*'*20)
    if cluster_resolver.task_type in ("worker", "ps", "chief"):
        os.environ["GRPC_FAIL_FAST"] = "use_caller"

        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol=cluster_resolver.rpc_layer or "grpc",
            start=True)
        server.join()
    # strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

if __name__ == '__main__':
    print(time.time())
    args = parse_args()
    main(args)