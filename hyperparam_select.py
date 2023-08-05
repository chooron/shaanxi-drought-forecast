import pandas as pd
import os

from darts import TimeSeries

from constant import METHOD_DICT, MODEL_DICT

area_id = 1
monthly = 1
method = 'VMD'
model_nm = 'LSTM'


def objective(trial):
    df = pd.read_csv(os.path.join('cache', 'spei', f'shp_{area_id}', f'spei_{monthly}', f'{method}_dec.csv'))
    df['DATE'] = pd.date_range('19790101', '20190101', freq='m')[monthly - 1:]
    target = TimeSeries.from_dataframe(df, time_col='DATE', value_cols='ORIG')
    series = TimeSeries.from_dataframe(df, time_col='DATE', value_cols=METHOD_DICT[method])
    train_series, test_series = series[:int(0.8 * len(series))], series[int(0.8 * len(series)) - 12:]
    train_series, test_series = series[:int(0.8 * len(series))], series[int(0.8 * len(series)) - 12:]
    train_target, test_target = target[:int(0.8 * len(target))], target[int(0.8 * len(target)) - 12:]

    model = MODEL_DICT[model_nm]
    trainer = pl.Trainer(enable_checkpointing=True,
                         callbacks=[ModelCheckpoint(dirpath=os.path.join('checkpoint', f'shp-{area_id}',
                                                                         f'spei-{monthly}', 'DLinear')),
                                    EarlyStopping(patience=10, monitor='val_loss')])
    model.fit(series=train_target, past_covariates=train_series, trainer=trainer)
    pred = model.predict(n=len(test_series) - 12, series=train_target,
                         past_covariates=test_series)
    train_pred = model.historical_forecasts(series=train_target,
                                            past_covariates=train_series,
                                            start=12, stride=1,
                                            retrain=False)
