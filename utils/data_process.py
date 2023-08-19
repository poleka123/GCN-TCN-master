import numpy as np
import pandas as pd
from utils.utils import Z_Score, generate_dataset, warm_PM_paramters_perK

def load_matrix(file_path):
    return pd.read_csv(file_path, header=None).to_numpy(np.float32)


def Data_Process(data_path, timesteps_input, timesteps_output, fftwarmlen = 24*30*2, K=100, J=10):
    data_values = pd.read_csv(data_path, header=None).head(8640).to_numpy(np.float32)
    # normalization
    data_values = np.reshape(data_values, (data_values.shape[0], data_values.shape[1], 1)).transpose((1, 2, 0))
    data_values, data_mean, data_std = Z_Score(data_values)
    # data split
    index_1 = int(data_values.shape[2] * 0.8)
    train_original_data = data_values[:, :, :index_1]
    val_original_data = data_values[:, :, index_1:]
    #FFT
    fft_warm_point = train_original_data.shape[2] - fftwarmlen
    time_series = np.reshape(train_original_data, (train_original_data.shape[0], train_original_data.shape[2]))
    all_a, all_p, all_f, all_mean, all_Kmask = warm_PM_paramters_perK(time_series, fft_warm_point, K, J)

    # generate_data
    train_input, train_target, train_input_time, train_target_time, ftmask, tmask = generate_dataset(train_original_data,
                                                 num_timesteps_input=timesteps_input,
                                                 num_timesteps_output=timesteps_output)
    evaluate_input, evaluate_target, evaluate_input_time, evaluate_target_time, evmask, emask = generate_dataset(val_original_data,
                                                       num_timesteps_input=timesteps_input,
                                                       num_timesteps_output=timesteps_output)

    data_set = {
        'train_input': train_input,
        'train_target': train_target,
        'train_input_time': train_input_time,
        'train_target_time': train_target_time,
        'tfmask': ftmask,
        'tmask': tmask,
        'eval_input': evaluate_input,
        'eval_target': evaluate_target,
        'eval_input_time': evaluate_input_time,
        'eval_target_time': evaluate_target_time,
        'efmask': evmask,
        'etmask': emask,
        'data_mean': data_mean,
        'data_std': data_std
    }
    return data_set, all_a, all_p, all_f, all_mean, all_Kmask

