import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Настройка стиля графиков для диплома (академический стиль)
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(filepath):
    print(f"Загрузка данных из {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Размер датасета: {df.shape}")
    return df

def plot_class_distribution(df):
    """
    1. Визуализация дисбаланса классов (Log scale)
    Используем логарифмическую шкалу, иначе столбец Fraud будет невидимым.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Class', data=df, palette=['#1f77b4', '#d62728'])
    
    plt.title('Распределение классов транзакций (логарифмическая шкала)', fontsize=14)
    plt.xlabel('Класс (0: Легитимные, 1: Мошеннические)', fontsize=12)
    plt.ylabel('Количество транзакций (log)', fontsize=12)
    
    # Включаем логарифмическую шкалу по оси Y
    plt.yscale('log')
    
    # Добавляем подписи значений над столбцами
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)
    print("График class_distribution.png сохранен.")
    plt.show()

def plot_amount_distribution(df):
    """
    2. Анализ распределения сумм (Boxplot)
    Показывает разницу в медианах и наличие выбросов.
    """
    plt.figure(figsize=(10, 6))
    
    # Добавляем 1e-5, чтобы избежать log(0), если есть нулевые суммы
    df['Amount_Log'] = np.log1p(df['Amount'])
    
    sns.boxplot(x='Class', y='Amount', data=df, palette=['#1f77b4', '#d62728'], showfliers=True)
    
    plt.title('Распределение сумм транзакций по классам', fontsize=14)
    plt.xlabel('Класс', fontsize=12)
    plt.ylabel('Сумма транзакции (€)', fontsize=12)
    plt.yscale('log') # Логарифмическая шкала для Y, чтобы видеть разброс
    
    plt.tight_layout()
    plt.savefig('amount_boxplot.png', dpi=300)
    print("График amount_boxplot.png сохранен.")
    plt.show()

def plot_time_distribution(df):
    """
    3. Анализ временной структуры (KDE Plot)
    Сравниваем плотность распределения во времени.
    """
    plt.figure(figsize=(12, 6))
    
    # Разделяем классы
    class_0 = df[df['Class'] == 0]['Time']
    class_1 = df[df['Class'] == 1]['Time']
    
    # Переводим секунды в часы (для наглядности на графике)
    # 48 часов = 172800 секунд
    sns.kdeplot(class_0 / 3600, label='Легитимные (Class 0)', shade=True, color='#1f77b4')
    sns.kdeplot(class_1 / 3600, label='Мошеннические (Class 1)', shade=True, color='#d62728')
    
    plt.title('Плотность распределения транзакций во времени', fontsize=14)
    plt.xlabel('Время (часы с начала сбора данных)', fontsize=12)
    plt.ylabel('Плотность вероятности', fontsize=12)
    plt.xlim([0, 48])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_kde.png', dpi=300)
    print("График time_kde.png сохранен.")
    plt.show()

def plot_correlation_matrix(df):
    """
    4. Матрица корреляций
    """
    plt.figure(figsize=(14, 10))
    
    # Считаем корреляцию Пирсона
    corr_matrix = df.corr()
    
    # Рисуем хитмап
    sns.heatmap(corr_matrix, cmap='coolwarm_r', annot=False, fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    plt.title('Матрица корреляций признаков', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    print("График correlation_matrix.png сохранен.")
    plt.show()

    # Дополнительно: Вывод топ корреляций с классом Class
    print("\nТоп признаков, коррелирующих с мошенничеством (Class):")
    print(corr_matrix['Class'].sort_values(ascending=False).head(5))
    print(corr_matrix['Class'].sort_values(ascending=True).head(5))

# --- ЗАПУСК ---
if __name__ == "__main__":
    # Замени путь на свой
    file_path = 'creditcard.csv' 
    
    try:
        df = load_data(file_path)
        
        plot_class_distribution(df)
        plot_amount_distribution(df)
        plot_time_distribution(df)
        plot_correlation_matrix(df)
        
        print("\nВсе графики успешно построены и сохранены.")
        
    except FileNotFoundError:
        print("Ошибка: Файл creditcard.csv не найден. Пожалуйста, скачай его с Kaggle.")