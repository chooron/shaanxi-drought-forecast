import os

import pandas as pd
import xarray as xr

# 读取气象数据
base_path = r'D:\code\py\pycharm\My Project\Some Idea\shaanxi-drought-forecast'
for i in range(1, 13):
    temp_dict = []
    for var in ['gamma_01', 'gamma_03', 'gamma_06', 'gamma_09', 'gamma_12']:
        ds = xr.open_dataset(os.path.join(base_path, 'drought', 'spei', f'shp_{i}', f'YunLin_spei_{var}.nc'))
        temp_dict.append(ds.mean(dim=['lat', 'lon']).to_dataframe())
    pd.concat(temp_dict, axis=1).to_csv(os.path.join(base_path, 'output', 'spei', f'shp_{i}_spei_gamma_monthly.csv'))
