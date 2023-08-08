import pandas as pd
import numpy as np
import os
import json
from darts import TimeSeries, concatenate
from darts.models import DLinearModel, LightGBMModel
from darts.metrics import r2_score, mape, mae, rmse, mse
from darts.dataprocessing.transformers import Scaler

from constant import get_trainer, EPOCH, get_model
import warnings

warnings.filterwarnings("ignore")
look_back = 12


def save_result(target, pred, save_path, type):
    save_path = os.path.join(save_path, type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_compare = concatenate([target, pred], axis=1)
    train_compare_df = train_compare.pd_dataframe()
    train_compare_df.columns = ['real', 'pred']
    train_compare_df.to_csv(os.path.join(save_path, f'results.csv'))
    criteria_dict = {'r2': str(r2_score(target, pred)),
                     'mape': str(mape(target, pred)),
                     'mae': str(mae(target, pred)),
                     'rmse': str(rmse(target, pred)),
                     'mse': str(mse(target, pred))}
    with open(os.path.join(save_path, f'criteria.json'), 'w') as f:
        json.dump(criteria_dict, f)


def main(area_id, monthly, model_nm, method):
    """
    预测主程序
    :param area_id: 区域id
    :param monthly: spei的尺度
    :param model_nm: 模型名称
    :param method: 分解方法名称
    :return:
    """

    checkpoint_path = os.path.join('checkpoint', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')
    log_path = os.path.join('log', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')
    save_path = os.path.join('save', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')

    if os.path.exists(checkpoint_path):
        return None

    if method == 'None':
        df = pd.read_csv(os.path.join('cache', 'YunLin', 'spei', f'shp_{area_id}', f'spei_{monthly}', f'VMD_dec.csv'))[
            ['ORIG']]
    else:
        df = pd.read_csv(
            os.path.join('cache', 'YunLin', f'shp_{area_id}', f'spei_{monthly}', f'{method}_dec.csv'))
    series_cols = df.columns.values.tolist()
    df['DATE'] = pd.date_range('19790101', '20190101', freq='m')[monthly - 1:]

    target = TimeSeries.from_dataframe(df, time_col='DATE', value_cols='ORIG').astype(np.float32)
    series = TimeSeries.from_dataframe(df, time_col='DATE', value_cols=series_cols).astype(np.float32)
    train_series, val_series, test_series = (series[:int(0.8 * len(series))],
                                             series[int(0.8 * len(series)) - look_back:int(0.9 * len(series))],
                                             series[int(0.9 * len(series)) - look_back:])
    train_target, val_target, test_target = (target[:int(0.8 * len(target))],
                                             target[int(0.8 * len(target)) - look_back:int(0.9 * len(target))],
                                             target[int(0.9 * len(target)) - look_back:])
    series_scaler = Scaler().fit(train_series)
    train_series_scaled = series_scaler.transform(train_series)
    val_series_scaled = series_scaler.transform(val_series)
    test_series_scaled = series_scaler.transform(test_series)

    target_scaler = Scaler().fit(train_target)
    train_target_scaled = target_scaler.transform(train_target)
    val_target_scaled = target_scaler.transform(val_target)
    # test_target_scaled = target_scaler.transform(test_target)

    model: LightGBMModel = get_model(model_nm)
    if model_nm in ['LightGBM', 'XBGoost']:
        model.fit(series=train_target_scaled, past_covariates=train_series_scaled,
                  val_series=val_target_scaled, val_past_covariates=val_series_scaled)
    else:
        trainer = get_trainer(checkpoint_path, log_path)
        # torch model fit
        model.fit(series=train_target_scaled, past_covariates=train_series_scaled, trainer=trainer,
                  val_series=val_target_scaled, val_past_covariates=val_series_scaled, epochs=EPOCH)
    model.fit(series=train_target_scaled, past_covariates=train_series_scaled,
              val_series=val_target_scaled, val_past_covariates=val_series_scaled)
    pred = model.predict(n=len(test_series) - look_back, series=val_target_scaled,
                         past_covariates=test_series_scaled)
    train_pred = model.historical_forecasts(series=train_target_scaled,
                                            past_covariates=train_series_scaled,
                                            start=look_back, stride=1, verbose=True,
                                            retrain=False, show_warnings=False)
    val_pred = model.historical_forecasts(series=val_target_scaled,
                                          past_covariates=val_series_scaled,
                                          start=look_back, stride=1, verbose=True,
                                          retrain=False, show_warnings=False)
    save_result(train_target[12:], target_scaler.inverse_transform(train_pred), save_path, 'train')
    save_result(val_target[12:], target_scaler.inverse_transform(val_pred), save_path, 'val')
    save_result(test_target[12:], target_scaler.inverse_transform(pred), save_path, 'test')


if __name__ == '__main__':
    for i in range(1, 13):
        for j in [1, 3, 6, 9, 12]:
            for model_nm in ['LightGBM', 'XBGoost']:
                for method in ['EEMD', 'VMD' ,'None']:  #
                    main(i, j, model_nm, method)
