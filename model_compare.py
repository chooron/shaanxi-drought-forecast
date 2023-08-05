import pandas as pd
import numpy as np
import os
import json
from darts import TimeSeries, concatenate
from darts.models import DLinearModel
from darts.metrics import r2_score, mape, mae, rmse, mse

from constant import get_trainer, EPOCH, get_model

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
        json.dumps(criteria_dict)


def main(area_id, monthly, model_nm, method):
    checkpoint_path = os.path.join('checkpoint', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')
    log_path = os.path.join('log', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')
    save_path = os.path.join('save', f'shp-{area_id}', f'spei-{monthly}', f'{model_nm}', f'{method}')

    df = pd.read_csv(os.path.join('cache', 'spei', f'shp_{area_id}', f'spei_{monthly}', f'{method}_dec.csv'))
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

    model: DLinearModel = get_model(model_nm)
    trainer = get_trainer(checkpoint_path, log_path)
    model.fit(series=train_target, past_covariates=train_series, trainer=trainer,
              val_series=val_target, val_past_covariates=val_series, epochs=EPOCH)
    pred = model.predict(n=len(test_series) - look_back, series=val_target,
                         past_covariates=test_series)
    train_pred = model.historical_forecasts(series=train_target,
                                            past_covariates=train_series,
                                            start=look_back, stride=1,
                                            retrain=False, show_warnings=False)
    val_pred = model.historical_forecasts(series=val_target,
                                          past_covariates=val_series,
                                          start=look_back, stride=1,
                                          retrain=False, show_warnings=False)
    save_result(train_target[12:], train_pred, save_path, 'train')
    save_result(val_target[12:], val_pred, save_path, 'val')
    save_result(test_target[12:], pred, save_path, 'test')


if __name__ == '__main__':
    for i in range(1, 13):
        for j in [1, 3, 6, 9, 12]:
            for model_nm in ['LSTM', 'RNN', 'TCN']:
                for method in ['VMD', 'EEMD', 'CEEMD']:  #
                    main(i, j, model_nm, method)
