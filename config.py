import argparse

criteo_dims = [7912889,33823,582469,245828,1,2209,10667,104,4,968,15,
                8165896,17139,2675940, 7156453,302516,12022,97,
                35,7339,20046,4,7105,1382,63,5554114]

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    # model_config
    parser.add_argument('--model_type', type=str, default='dcnv2', help='dcnv2 or fwfm')
    parser.add_argument('--data_path', type=str, default='/data')
    parser.add_argument('--embedding_dims', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=6400, help='batch size per gpu')
    parser.add_argument('--dnn_layers', nargs='+', type=int, default=[1024, 512, 256, 128])
    parser.add_argument('--num_gpus', type=int, default=1, help='total num of gpus')
    parser.add_argument('--num_batch', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    
    parser.add_argument('--sparse_feature_cardinaltity', nargs='+', default= criteo_dims, help='sparse feature dims')
    parser.add_argument('--embedding_hash_size', type=int, default=1000)

    # parameter_server
    parser.add_argument('--use_ps', type=int, default=0, help='0 for False and 1 for True')
    parser.add_argument('--job_name', type=str,default='ps', help='ps, worker or chief')
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--num_ps', type=int, default=2)

    return parser.parse_args()
