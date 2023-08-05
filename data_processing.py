# 将各地区的干旱序列进行分解

import pandas as pd
import os
from utils.decompose import decompose_method_dict

for i in range(1, 13):
    df = pd.read_csv(os.path.join('data', 'spei', f'shp_{i}_spei_gamma_monthly.csv'))
    for j in [1, 3, 6, 9, 12]:
        series = df.iloc[j - 1:, :][f'spei_gamma_{str(j).zfill(2)}'].values
        for method in ['CEEMD']:
            dec_df = decompose_method_dict[method](series, window=10)
            save_path = os.path.join('cache', 'spei', f'shp_{i}', f'spei_{j}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dec_df.to_csv(os.path.join(save_path, f'{method}_dec.csv'), index=False)

