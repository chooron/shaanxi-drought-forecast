"""
@author jingxin
基于ssa和vmd分解,滑动拼接训练、开发和测试,生成训练样本、开发样本i和测试样本i
"""
from utils.LMD import lmd_main
# 准备: pip install EMD-signal 安装EMD
from utils.VMD import VMD
from utils.SSA import SSA
import pandas as pd
import numpy as np
import os


def execute_ssa(data, window):
    """
    SSA 分解包装语句
    :param data: 输入序列
    :param window: 分解阶数
    :return: 分解结果
    """
    reconstruct_ssa = SSA(data, window)
    orig_TS = reconstruct_ssa.orig_TS.values
    result = np.array(orig_TS)
    columns = ['ORIG', 'TREND']
    for i in range(window):
        result = np.vstack((result, np.array(reconstruct_ssa.reconstruct(i).values)))
        if i > 0:
            columns = columns + ['Periodic' + str(i - 1)]
    array = result.T
    df_ssa = pd.DataFrame(array, columns=columns)
    return df_ssa


def execute_vmd(data, window, alpha=2000, tau=0.0, DC=0, init=1, tol=1e-9):
    """
    VMD 分解包装语句
    :param data: 输入序列
    :param window: 分解阶数
    :return: 分解结果
    """
    vmd_df, _, _ = VMD(data, alpha, tau, window, DC, init, tol)
    vmd_df = np.concatenate([data.reshape(-1, 1), vmd_df], axis=1)
    columns = ['ORIG', 'TREND']
    for i in range(window - 1):
        columns = columns + ['Imf_' + str(i)]
    df_vmd = pd.DataFrame(vmd_df, columns=columns)
    return df_vmd


def execute_emd(data, window):
    """
    EMD 分解包装语句
    :param data: 输入序列
    :param window: 分解阶数
    :return: 分解结果
    """
    from PyEMD import EMD
    decomposer = EMD()
    emd_output = decomposer.emd(data, max_imf=window).T
    emd_output = np.concatenate([data.reshape(-1, 1), emd_output], axis=1)
    columns = ['ORIG', 'TREND']
    for i in range(emd_output.shape[1] - 2):
        columns = columns + ['Imf_' + str(i)]
    df_emd = pd.DataFrame(emd_output, columns=columns)
    return df_emd


def execute_eemd(data, window):
    """
    EEMD 分解包装语句
    :param data: 输入序列
    :param window: 分解阶数
    :return: 分解结果
    """
    from PyEMD import EEMD
    decomposer = EEMD(trials=100)
    eemd_output = decomposer.eemd(data, max_imf=window).T
    eemd_output = np.concatenate([data.reshape(-1, 1), eemd_output], axis=1)
    columns = ['ORIG', 'TREND']
    for i in range(eemd_output.shape[1] - 2):
        columns = columns + ['Imf_' + str(i)]
    df_eemd = pd.DataFrame(eemd_output, columns=columns)
    return df_eemd


def execute_ceemd(data, window):
    """
    EEMD 分解包装语句
    :param data: 输入序列
    :param window: 分解阶数
    :return: 分解结果
    """
    from PyEMD import CEEMDAN
    decomposer = CEEMDAN(trials=100)
    eemd_output = decomposer.ceemdan(data, max_imf=window).T
    eemd_output = np.concatenate([data.reshape(-1, 1), eemd_output], axis=1)
    columns = ['ORIG', 'TREND']
    for i in range(eemd_output.shape[1] - 2):
        columns = columns + ['Imf_' + str(i)]
    df_eemd = pd.DataFrame(eemd_output, columns=columns)
    return df_eemd


def exectue_lmd(data, window=0):
    A, S, PF, remnant = lmd_main(data, np.linspace(0, len(data), num=len(data)), simple='yes', len_pf=window)
    columns = [f'PF_{i}' for i in range(1, PF.shape[1] + 1)]
    PF_df = pd.DataFrame(PF, columns=columns)
    PF_df['REMNANT'] = remnant
    PF_df['ORIG'] = data
    return PF_df


decompose_method_dict = {'SSA': execute_ssa, 'VMD': execute_vmd, 'EEMD': execute_eemd, 'EMD': execute_emd,
                         'CEEMD': execute_ceemd}


def gen_direct_samples(input_df, output_df, lags_dict, lead_time):
    """
    根据输入表格和输出表格构建预测样本
    :param input_df: 输入特征
    :param output_df: 预测目标
    :param lags_dict: 各特征的滞后长度
    :param lead_time: 预见期
    :return: 预测样本
    """
    max_lag = max(lags_dict.values())
    input_columns = list(input_df.columns)
    # Get the number of input features
    signals_num = input_df.shape[1]
    # Get the data size
    data_size = input_df.shape[0]
    # Compute the samples size
    samples_size = data_size - max_lag
    # Generate input colmuns for each input feature
    samples = pd.DataFrame()
    for i in range(signals_num):
        one_in = (input_df[input_columns[i]]).values  # subsignal
        lag = lags_dict[input_columns[i]]
        oness = pd.DataFrame()  # restor input features
        for j in range(lag):
            x = pd.DataFrame(one_in[j:data_size - (lag - j)],
                             columns=[input_columns[i] + '{t-' + str(lag - j - 1) + '}'])
            oness = pd.concat([oness, x], axis=1, sort=False)
        oness = oness.iloc[oness.shape[0] - samples_size:]
        oness = oness.reset_index(drop=True)
        samples = pd.concat([samples, oness], axis=1, sort=False)
    target = output_df[max_lag + lead_time - 1:]
    time_index = target.index
    target = pd.DataFrame(target.values, columns=[list(output_df.columns)[0] + '{t+' + str(lead_time) + '}'])
    samples = samples[:samples.shape[0] - (lead_time - 1)]
    samples = pd.concat([samples, target], axis=1)
    samples = samples.set_index(time_index, drop=True)
    return samples


def decompose_processing(df, window, decompose_method, sample_config, output_path, lead_time, lag_list=None):
    """
    分解主程序
    :param df: 待分解序列
    :param window: 分解阶数
    :param decompose_method: 分解方法
    :param sample_config: 训练测试验证比例
    :param output_path: 分解结果输出路径
    :param lead_time: 预见期
    :param lag_list: 分解子序列滞后
    :return:  训练测试验证分解结果
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_len = int(sample_config['train_prop'] * len(df))
    dev_len = int(sample_config['dev_prop'] * len(df))
    test_len = len(df) - dev_len - train_len
    train = df[:train_len]
    train_df = decompose_method_dict[decompose_method](train, window)
    train_sample_df = gen_direct_samples(train_df, pd.DataFrame(train, columns=['ORIG']),
                                         lag_select(train_df, lag_list), lead_time)
    train_sample_df.to_csv(output_path + 'train_decompose_sample.csv', index=False)
    dev_sample_df = pd.DataFrame(columns=train_sample_df.columns)
    test_sample_df = pd.DataFrame(columns=train_sample_df.columns)
    # 附加分解步骤
    for i in range(dev_len):
        print('dev: 第' + str(i) + '/' + str(dev_len) + '个添加开始')
        dev_i_df = df[:train_len + i]
        dev_i_df = decompose_method_dict[decompose_method](dev_i_df, window)
        dev_sample_i_df = gen_direct_samples(dev_i_df, pd.DataFrame(dev_i_df, columns=['ORIG']),
                                             lag_select(dev_i_df, lag_list), lead_time)
        dev_sample_df = dev_sample_df.append(dev_sample_i_df.iloc[-1], ignore_index=True)
    for i in range(test_len):
        print('test: 第' + str(i) + '/' + str(test_len) + '个添加开始')
        test_i_df = df[:train_len + dev_len + i]
        test_i_df = decompose_method_dict[decompose_method](test_i_df, window)
        test_sample_i_df = gen_direct_samples(test_i_df, pd.DataFrame(test_i_df, columns=['ORIG']),
                                              lag_select(test_i_df, lag_list), lead_time)
        test_sample_df = test_sample_df.append(test_sample_i_df.iloc[-1], ignore_index=True)
    dev_sample_df.to_csv(output_path + 'dev_sample.csv', index=False)
    test_sample_df.to_csv(output_path + 'test_sample.csv', index=False)
    return train_sample_df, dev_sample_df, test_sample_df


def lag_select(df, lag_list=None):
    """
    设置滞后长度
    :param df: 需要滞后的表格
    :param lag_list: 各列的滞后长度
    :return: 各列的滞后长度字典格式
    """
    if lag_list is None:
        lag_list = [4] * len(df.columns)
    return {c: lag_list[i] for i, c in enumerate(df.columns)}


if __name__ == '__main__':
    df1 = pd.read_csv(
        r'D:\code\py\pycharm\My Project\My Lib\hydrology-forecast\project\drought-forecast\data\spei\shp_1_spei_gamma_monthly.csv')[
        'spei_gamma_01'].to_numpy().reshape(-1, 1)
    df = exectue_lmd(df1, 4)
    # result = decompose_processing(df1,
    #                               sample_config={'train_prop': 0.8, 'dev_prop': 0.1, 'test_prop': 0.1},
    #                               decompose_method='EEMD',
    #                               output_path=r'F:\pycharm\Some code\forecast\AdaptiveStreamflowForecastService\sample',
    #                               window=8,
    #                               lead_time=1)
