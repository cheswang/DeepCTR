import pandas as pd

def read_data(args):
    chunksize = args.batch_size
    file_path = args.data_path + '/day_0'
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I'+str(i) for i in range(1, 14)]
    columns = ['label'] + dense_features + sparse_features
    # 读取文件并将其转换为Pandas DataFrame对象
    data = pd.read_csv(file_path, sep='\t', iterator=True, chunksize=chunksize, names=columns)

    return data