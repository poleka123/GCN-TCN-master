import os
import argparse
import torch
import torch.nn as nn
from utils.data_load import Data_load
from utils.data_process import Data_Process, load_matrix
from utils.utils import *
from methods.train import Train
from methods.evaluate import Evaluate
import logger
# from model.TreeCNs import TreeCNs
from model.GassConv import GTCN, PeriodicityModule
import numpy as np

torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 32)')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=64)
parser.add_argument('--spatial_channels', type=int, default=16)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--time_slice', type=list, default=[1, 2, 3])

args = parser.parse_args()


def gauss(kernel_size_x, kernel_size_y, sigma):
    kernel = np.zeros((kernel_size_x, kernel_size_y))
    center = kernel_size_y // 2
    if sigma <= 0:
        sigma = ((kernel_size_y - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size_x):
        for j in range(kernel_size_y):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


if __name__ == '__main__':

    torch.manual_seed(7)
    elogger = logger.Logger('run_log_gcn_tcn_2')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # NATree, data_set = Data_load(args.timesteps_input, args.timesteps_output)
    data_set, all_a, all_p, all_f, all_mean, all_Kmask = Data_Process("./data_set/SmallScaleAggregation/V_flow_50.csv",
                                                                      args.timesteps_input, args.timesteps_output,
                                                                      24 * 30 * 2, 128, 10)
    W_nodes = load_matrix('./data_set/SmallScaleAggregation/distance_50.csv')


    Ks = 3 #多项式近似个数
    NATree = np.ones((50,128))
    Num_of_nodes = NATree.shape[0]
    # MaxNodeNumber = NATree.shape[2]
    # MaxLayerNumber = NATree.shape[1]
    g_kernel = gauss(1, args.ksize, 0.5)
    input_channels = 1
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    ids = torch.from_numpy(np.arange(0, Num_of_nodes)).to(device)
    all_Kmask = torch.tensor(all_Kmask, dtype=torch.float32).to(device)
    FFT_data = {
        'all_a': all_a,
        'all_p': all_p,
        'all_f': all_f,
        'all_mean': all_mean
    }
    data_set1 = data_set
    data_set1['all_Kmask'] = all_Kmask
    data_set1['ids'] = ids
    # 切比雪夫多项式近似
    L = scaled_laplacian(W_nodes)
    Lk = cheb_poly(L, Ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    # Gauss Conv Net
    model = GTCN(
        FFT_data,
        Ks,
        input_channels,
        args.timesteps_input,
        args.timesteps_output,
        channel_sizes,
        g_kernel,
        kernel_size=kernel_size,
        dropout=args.dropout,
        device=device
    )

    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        print("Train Process")
        train_loss = Train(
            model=model,
            optimizer=optimizer,
            loss_meathod=L2,
            NATree=Lk,
            data_set=data_set1,
            batch_size=args.batch_size,
            device=device
        )
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            eval_loss, eval_index = Evaluate(
                epoch=epoch,
                model=model,
                loss_meathod=L2,
                NATree=Lk,
                time_slice=args.time_slice,
                data_set=data_set1,
                device=device
            )
        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(train_loss))
        elogger.log("epoch:{}".format(epoch))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                          eval_index['MAE'][-(len(args.time_slice) - i)],
                          eval_index['RMSE'][-(len(args.time_slice) - i)],
                          eval_index['sMAPE'][-(len(args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
                                eval_index['MAE'][-(len(args.time_slice) - i)],
                                eval_index['RMSE'][-(len(args.time_slice) - i)],
                                eval_index['sMAPE'][-(len(args.time_slice) - i)]))
        elogger.log("---------------------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------------------")
