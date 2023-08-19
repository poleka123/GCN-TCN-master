import os
import argparse
import torch
import torch.nn as nn
from utils.data_load import Data_load
from utils.data_process import Data_Process
from utils.utils import *
from methods.train import Train
from methods.evaluate import Evaluate
import logger
# from model.TreeCNs import TreeCNs
from model.models import  PeriodicityModule, depts_expansion_general
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
    elogger = logger.Logger('run_log_dep')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # NATree, data_set = Data_load(args.timesteps_input, args.timesteps_output)
    data_set, all_a, all_p, all_f, all_mean, all_Kmask = Data_Process("./data_set/SmallScaleAggregation/V_flow_50.csv", args.timesteps_input, args.timesteps_output, 24*30*2, 128, 10)
    # print(data_set)
    # 50 nodes of ids [0, 1, 2, ... ,50]
    ids = torch.from_numpy(np.arange(0, 50)).to(device)
    num_of_nodes = 50
    all_Kmask = torch.tensor(all_Kmask, dtype=torch.float32).to(device)
    PM = PeriodicityModule(all_a, all_p, all_f, all_mean, device).to(device)
    generic_layer_size, generic_layers, generic_stacks = 64, 4, 15
    period_layers = 1
    EM = depts_expansion_general(input_size=args.features, output_size=args.timesteps_output,
                                 layer_size=generic_layer_size, stacks=generic_stacks, local_layers=generic_layers,
                                 period_layers=period_layers, num_series=num_of_nodes).to(device)
    # N = NATree.shape[0]
    # MaxNodeNumber = NATree.shape[2]
    # MaxLayerNumber = NATree.shape[1]
    # g_kernel = gauss(1, args.ksize, 0.5)
    # input_channels = 1
    # channel_sizes = [args.nhid] * args.levels
    # kernel_size = args.ksize
    # # Gauss Conv Net
    # model = GTCN(
    #     input_channels,
    #     args.timesteps_input,
    #     args.timesteps_output,
    #     channel_sizes,
    #     g_kernel,
    #     kernel_size=kernel_size,
    #     dropout=args.dropout
    # )
    #
    #
    # if torch.cuda.is_available():
    #     model.cuda()
    #     NATree = torch.from_numpy(NATree).cuda()
    PM_optimizer = torch.optim.Adam(PM.parameters(), lr=0.001)
    EM_optimizer = torch.optim.Adam(EM.parameters(), lr=0.001)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        print("Train Process")
        permutation = torch.randperm(data_set['train_input'].shape[0])
        epoch_training_losses = []
        loss_mean = 0.0
        # train
        for i in range(0, data_set['train_input'].shape[0], args.batch_size):
            PM.train()
            EM.train()
            PM_optimizer.zero_grad()
            EM_optimizer.zero_grad()
            indices = permutation[i:i+args.batch_size]
            X_batch, y_batch = data_set['train_input'][indices], data_set['train_target'][indices]
            X_mask, y_mask = data_set['tfmask'][indices], data_set['tmask'][indices]
            X_timestamp, y_timestamp = data_set['train_input_time'][indices], data_set['train_target_time'][indices]

            if torch.cuda.is_available():
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                X_mask = X_mask.to(device)
                y_mask = y_mask.to(device)
                X_timestamp = X_timestamp.to(device)
                y_timestamp = y_timestamp.to(device)
                ids = ids.to(device)
                std = torch.tensor(data_set['data_std']).to(device)
                mean = torch.tensor(data_set['data_mean']).to(device)
            else:
                std = torch.tensor(data_set['data_std'])
                mean = torch.tensor(data_set['data_mean'])
            X_z, y_z = PM(X_timestamp, ids.long(), all_Kmask), PM(y_timestamp, ids.long(), all_Kmask)
            perd, _, _ = EM(X_batch, X_z, X_mask, y_z, ids)
            perd, y_batch = Un_Z_Score(perd, mean, std), Un_Z_Score(y_batch, mean, std)
            loss = L2(perd, y_batch)
            loss.backward()
            PM_optimizer.step()
            EM_optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
            loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
            if i % 50 == 0:
                print("Loss Mean "+str(loss_mean))

        torch.cuda.empty_cache()
        PM.eval()
        EM.eval()
        with torch.no_grad():
            eval_input = data_set['eval_input']
            eval_target = data_set['eval_target']
            X_mask, y_mask = data_set['efmask'], data_set['etmask']
            X_timestamp, y_timestamp = data_set['eval_input_time'], data_set['eval_target_time']
            if torch.cuda.is_available():
                eval_input = eval_input.to(device)
                eval_target = eval_target.to(device)
                X_mask = X_mask.to(device)
                y_mask = y_mask.to(device)
                X_timestamp = X_timestamp.to(device)
                y_timestamp = y_timestamp.to(device)
                ids = ids.to(device)
                std = torch.tensor(data_set['data_std']).to(device)
                mean = torch.tensor(data_set['data_mean']).to(device)
            else:
                std = torch.tensor(data_set['data_std'])
                mean = torch.tensor(data_set['data_mean'])
            
            X_z, y_z = PM(X_timestamp, ids.long(), all_Kmask), PM(y_timestamp, ids.long(), all_Kmask)
            window_forecast, window_global, window_local = EM(eval_input, X_z, X_mask, y_z, ids)
            val_index = {}
            val_index['MAE'] = []
            val_index['RMSE'] = []
            val_index['sMAPE'] = []
            val_loss = []

            for item in args.time_slice:
                pred_index = window_forecast[:, :, item - 1]
                val_target_index = eval_target[:, :, item - 1]
                pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean,
                                                                                             std)

                loss = L2(pred_index, val_target_index)
                val_loss.append(loss)

                filePath = "./results/DEP/"
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                if ((epoch + 1) % 50 == 0) & (epoch != 0) & (epoch > 200):
                    np.savetxt(filePath + "/pred_" + str(epoch) + ".csv", pred_index.cpu(), delimiter=',')
                    np.savetxt(filePath + "/true_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')
                mae = MAE(val_target_index, pred_index)
                val_index['MAE'].append(mae)

                rmse = RMSE(val_target_index, pred_index)
                val_index['RMSE'].append(rmse)

                smape = SMAPE(val_target_index, pred_index)
                val_index['sMAPE'].append(smape)

        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(loss_mean))
        for i in range(len(args.time_slice)):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(args.time_slice[i] * 5, val_loss[-(len(args.time_slice) - i)],
                          val_index['MAE'][-(len(args.time_slice) - i)],
                          val_index['RMSE'][-(len(args.time_slice) - i)],
                          val_index['sMAPE'][-(len(args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(args.time_slice[i] * 5, val_loss[-(len(args.time_slice) - i)],
                                val_index['MAE'][-(len(args.time_slice) - i)],
                                val_index['RMSE'][-(len(args.time_slice) - i)],
                                val_index['sMAPE'][-(len(args.time_slice) - i)]))
        print("---------------------------------------------------------------------------------------------------")

        # train_loss = Train(
        #     model=PM,
        #     optimizer=optimizer,
        #     loss_meathod=L2,
        #     NATree=NATree,
        #     data_set=data_set,
        #     batch_size=args.batch_size
        # )
        # torch.cuda.empty_cache()
        # with torch.no_grad():
        #     print("Evalution Process")
        #     eval_loss, eval_index = Evaluate(
        #         epoch=epoch,
        #         model=model,
        #         loss_meathod=L2,
        #         NATree=NATree,
        #         time_slice=args.time_slice,
        #         data_set=data_set,
        #     )
        # print("---------------------------------------------------------------------------------------------------")
        # print("epoch: {}/{}".format(epoch, args.epochs))
        # print("Training loss: {}".format(train_loss))
        # for i in range(len(args.time_slice)):
        #     print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
        #           .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
        #                   eval_index['MAE'][-(len(args.time_slice) - i)],
        #                   eval_index['RMSE'][-(len(args.time_slice) - i)],
        #                   eval_index['sMAPE'][-(len(args.time_slice) - i)]))
        #     elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
        #                 .format(args.time_slice[i] * 5, eval_loss[-(len(args.time_slice) - i)],
        #                         eval_index['MAE'][-(len(args.time_slice) - i)],
        #                         eval_index['RMSE'][-(len(args.time_slice) - i)],
        #                         eval_index['sMAPE'][-(len(args.time_slice) - i)]))
        # print("---------------------------------------------------------------------------------------------------")
