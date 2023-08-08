import os

import pandas as pd
import xarray as xr

from constant import CITY_NAME

# 读取干旱数据结果
base_path = r'D:\code\py\pycharm\My Project\Some Idea\shaanxi-drought-forecast'
for f in os.listdir(os.path.join(base_path, 'nc', 'spei')):
    temp_dict = []
    for var in ['gamma_01', 'gamma_03', 'gamma_06', 'gamma_09', 'gamma_12']:
        ds = xr.open_dataset(os.path.join(base_path, 'nc', 'spei', f, f'_spei_{var}.nc'))
        temp_dict.append(ds.mean(dim=['lat', 'lon']).to_dataframe())
    pd.concat(temp_dict, axis=1).to_csv(
        os.path.join(base_path, 'data', 'shaanxi', 'spei', f'{f}_spei_gamma_monthly.csv'))

for c in CITY_NAME:
    temp_list = []
    for var in os.listdir(os.path.join(base_path, 'nc', 'met')):
        ds = xr.open_dataset(os.path.join(base_path, 'nc', 'met', var, f'{c}_{var}_monthly.nc'))
        temp_list.append(ds.mean(dim=['lat', 'lon']).to_dataframe())
    pd.concat(temp_list, axis=1).to_csv(
        os.path.join(base_path, 'data', 'shaanxi', 'met', f'{c}_met_monthly.csv'))