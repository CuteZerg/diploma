import pandas as pd
from sklearn.model_selection import train_test_split

print("Загрузка полных данных...")
df_orig = pd.read_csv('data/creditcard.csv')
df_proc = pd.read_parquet('data/creditcard_processed.parquet')

# Делаем точно такой же сплит, как при обучении (random_state=42, stratify)
print("Выделение тестовой выборки (56 962 транзакции)...")
y = df_proc['Class']
_, df_test = train_test_split(df_proc, test_size=0.2, random_state=42, stratify=y)

# Получаем индексы тестовой выборки, чтобы вытащить те же строки из оригинального CSV
test_indices = df_test.index

# Сбрасываем индексы, чтобы они совпадали в обоих файлах (от 0 до 56961)
df_test = df_test.reset_index(drop=True)
df_orig_test = df_orig.loc[test_indices].reset_index(drop=True)

# Сохраняем в папку demo_data
print("Сохранение...")
import os
os.makedirs('demo_data', exist_ok=True)
df_orig_test.to_csv('demo_data/creditcard_test.csv', index=False)
df_test.to_parquet('demo_data/creditcard_processed_test.parquet', index=False)

print("Готово! Данные для демо сохранены.")