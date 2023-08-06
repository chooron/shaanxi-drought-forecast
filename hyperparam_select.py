import pandas as pd
import numpy as np
import os
import optuna
import pytorch_lightning as pl

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel
from darts.metrics import rmse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

from constant import METHOD_DICT, BATCH_SIZE, OPTIM_CLS, OPTIM_KWARGS, EPOCH

area_id = 1
method = 'VMD'
model_nm = 'LSTM'


def main(monthly, method="VMD"):
    def objective(trial: optuna.Trial):
        look_back = trial.suggest_int('look_back', 4, 24, 4)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 256, 16)
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)

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

        model = BlockRNNModel(input_chunk_length=look_back, output_chunk_length=1, hidden_dim=hidden_dim,
                              model='LSTM', model_name='LSTM', batch_size=BATCH_SIZE, dropout=dropout,
                              show_warnings=True, optimizer_cls=OPTIM_CLS, optimizer_kwargs=OPTIM_KWARGS)

        trainer = pl.Trainer(enable_checkpointing=False,
                             logger=False,
                             callbacks=[
                                 # ModelCheckpoint(dirpath=os.path.join('checkpoint', f'shp-{area_id}',
                                 #                                      f'spei-{monthly}', 'DLinear')),
                                 EarlyStopping(patience=20, monitor='val_loss'),
                                 PyTorchLightningPruningCallback(monitor='val_los', trial=trial)
                             ])

        model.fit(series=train_target_scaled, past_covariates=train_series_scaled, trainer=trainer,
                  val_series=val_target_scaled, val_past_covariates=val_series_scaled, epochs=EPOCH)
        val_pred = model.historical_forecasts(series=val_target_scaled,
                                              past_covariates=val_series_scaled,
                                              start=look_back, stride=1, verbose=True,
                                              retrain=False, show_warnings=False)
        val_pred_rescaled = target_scaler.inverse_transform(val_pred)
        evaluate_result = rmse(val_target, val_pred_rescaled)
        return evaluate_result

    db_path = os.path.join('db', f'spei-{monthly}')
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    df = pd.read_csv(os.path.join('cache', 'spei', f'shp_{area_id}', f'spei_{monthly}', f'{method}_dec.csv'))
    df['DATE'] = pd.date_range('19790101', '20190101', freq='m')[monthly - 1:]
    target = TimeSeries.from_dataframe(df, time_col='DATE', value_cols='ORIG').astype(np.float32)
    series = TimeSeries.from_dataframe(df, time_col='DATE', value_cols=METHOD_DICT[method]).astype(np.float32)

    study = optuna.create_study(direction="minimize", storage=f'sqlite:///{db_path}/study.db', study_name=method,
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
    study_df = study.trials_dataframe(("number", "value", "duration",
                                       "params", "state"))
    study_df.to_csv(f'{db_path}/{method}-optimal.csv', index=False)


if __name__ == "__main__":
    for j in [1, 3, 6, 9, 12]:
        main(j)
