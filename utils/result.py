import os
import json
from darts import concatenate
from darts.metrics import r2_score, mape, mae, rmse, mse


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
