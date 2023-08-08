import pandas as pd


# df = pd.read_csv(r'yulin.csv')
# df = df[['shp', 'spei', 'model', 'dec', 'period', 'r2']]
# df = df[df['period'] == 'test']
# df['spei'] = [int(s.split('-')[1]) for s in df['spei'].values]
# df = df[df['model'] != 'TCN']
# df = df.drop('period', axis=1)
# pivot_df = df.pivot(index=['shp', 'spei', 'model'], columns=['dec'], values=['r2'])
# pivot_df.columns = [f'{level_0}-{level_1}' for level_0, level_1 in pivot_df.columns]
# pivot_df.to_csv('cache/analysis_result/r2_method_compare.csv')

def main(compare_obj, criterion):
    df = pd.read_csv(r'yulin.csv')
    df = df[['shp', 'spei', 'model', 'dec', 'period', 'r2']]
    df = df[df['period'] == 'test']
    df = df.drop('period', axis=1)
    df = df[df['dec'] != 'None']
    df['spei'] = [int(s.split('-')[1]) for s in df['spei'].values]
    df = df.drop('period', axis=1)
    pivot_df = df.pivot(index=['shp', 'spei', 'dec'], columns=['model'], values=['r2'])
    pivot_df.columns = [f'{level_0}-{level_1}' for level_0, level_1 in pivot_df.columns]
    pivot_df.to_csv('cache/analysis_result/r2_model_compare.csv')
