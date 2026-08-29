import pandas as pd

# Читаем большие оригинальные данные
df_orig = pd.read_csv('data/creditcard.csv')
df_proc = pd.read_parquet('data/creditcard_processed.parquet')

# Берем всех мошенников и 10 000 честных
fraud_idx = df_proc[df_proc['Class'] == 1].index
legit_idx = df_proc[df_proc['Class'] == 0].sample(10000, random_state=42).index
demo_idx = fraud_idx.union(legit_idx)

# Сохраняем мини-версии в папку demo_data
df_orig.loc[demo_idx].to_csv('demo_data/creditcard_demo.csv', index=False)
df_proc.loc[demo_idx].to_parquet('demo_data/creditcard_processed_demo.parquet', index=False)