import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Anti-Fraud System", layout="wide")
BASE_DIR = Path(__file__).resolve().parent.parent

# --- ЗАГРУЗКА РЕСУРСОВ ---
@st.cache_resource
def load_resources():
    model = xgb.XGBClassifier()
    model.load_model(BASE_DIR / "xgb_model.json")
    
    # Загружаем ТОЛЬКО правильную тестовую выборку (56 962 строки)
    df_processed = pd.read_parquet(BASE_DIR / "demo_data/creditcard_processed_test.parquet")
    df_orig = pd.read_csv(BASE_DIR / "demo_data/creditcard_test.csv")
    
    X_eval = df_processed.drop('Class', axis=1)
    y_eval = df_processed['Class'].values
    amounts_eval = df_orig['Amount'].values
    
    # Предрассчитываем вероятности для экономики на тесте
    probs_eval = model.predict_proba(X_eval)[:, 1]
    
    return model, df_processed, df_orig, y_eval, probs_eval, amounts_eval

model, df_processed, df_orig, y_eval, probs_eval, amounts_eval = load_resources()

# --- HEADER ---
st.title("Anti-Fraud Dashboard: ML в реальном бизнесе")
st.markdown("Дашборд демонстрирует применение XGBoost для выявления мошенничества в условиях экстремального дисбаланса (0.17%).")

# --- ВКЛАДКИ ---
tab1, tab2, tab3 = st.tabs(["Бизнес-Экономика и Порог", "Симулятор транзакций (What-If)", "Аналитика (EDA)"])

# ==========================================
# ВКЛАДКА 1: ЭКОНОМИКА И ПОРОГ
# ==========================================
with tab1:
    st.markdown("### Оптимизация порога: Математика (F1) против Бизнеса ($)")
    st.markdown("Модели машинного обучения оптимизируются по метрике **F1-score** (идеальный баланс между Precision и Recall). "
                "Но в бизнесе ложная тревога стоит $10 (звонок оператора), а пропущенный фрод может стоить тысячи долларов! "
                "Поэтому **финансовый оптимум** всегда отличается от алгоритмического.")
    
    # Ищем бизнес-оптимум на лету
    COST_OF_CALL = 10 
    thr_range = np.linspace(0.01, 0.99, 100)
    profits = []
    fps = []
    max_profit = -np.inf
    best_biz_thr = 0.5
    
    for t in thr_range:
        p = (probs_eval >= t).astype(int)
        sm = amounts_eval[(y_eval == 1) & (p == 1)].sum() # Поймали (Сохранили)
        ic = sum((y_eval == 0) & (p == 1)) * COST_OF_CALL # Ложные тревоги (Потратили)
        profit = sm - ic
        profits.append(profit)
        fps.append(sum((y_eval == 0) & (p == 1)))
        
        if profit > max_profit:
            max_profit = profit
            best_biz_thr = t

    col_slider, _ = st.columns([2, 1])
    with col_slider:
        threshold = st.slider("Ручная настройка порога блокировки", min_value=0.01, max_value=0.99, value=float(best_biz_thr), step=0.01)
    
    # Считаем метрики для выбранного порога
    preds = (probs_eval >= threshold).astype(int)
    total_fraud_amount = amounts_eval[y_eval == 1].sum()
    
    tp_mask = (y_eval == 1) & (preds == 1)
    fn_mask = (y_eval == 1) & (preds == 0)
    fp_mask = (y_eval == 0) & (preds == 1)
    
    saved_money = amounts_eval[tp_mask].sum()
    missed_money = amounts_eval[fn_mask].sum()
    investigation_cost = fp_mask.sum() * COST_OF_CALL
    
    # Общие финансовые потери банка (То, что мы хотим минимизировать)
    total_losses = missed_money + investigation_cost
    net_profit = saved_money - investigation_cost
    
    st.markdown("#### Финансовый срез на тестовой выборке (56 962 транзакции):")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Максимальный ущерб", f"${total_fraud_amount:,.0f}", help="Сумма всего фрода в выборке, если не использовать ML")
    col_m2.metric("Убытки банка с ML", f"${total_losses:,.0f}", delta=f"- Фрод: ${missed_money:,.0f} | Звонки: ${investigation_cost:,.0f}", delta_color="inverse")
    col_m3.metric("Предотвращенный ущерб", f"${saved_money:,.0f}", delta="Сохраненные средства")
    col_m4.metric("ЧИСТАЯ ВЫГОДА", f"${net_profit:,.0f}", delta="Макс. ущерб минус убытки с ML")
    
    st.markdown("---")
    st.markdown("**График: Поиск финансового оптимума (Где прибыль максимальна)**")
    
    fig_econ = go.Figure()
    fig_econ.add_trace(go.Scatter(x=thr_range, y=profits, name="Чистая выгода ($)", line=dict(color='green', width=3)))
    fig_econ.add_trace(go.Scatter(x=thr_range, y=fps, name="Кол-во ложных тревог (FP)", yaxis="y2", line=dict(color='red', width=2, dash='dot')))
    
    fig_econ.update_layout(
        xaxis=dict(title="Порог вероятности (Threshold)"),
        yaxis=dict(title="Чистая выгода ($)", title_font=dict(color="green")),
        yaxis2=dict(title="Ложные тревоги (шт)", title_font=dict(color="red"), anchor="x", overlaying="y", side="right"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
        hovermode="x unified"
    )
    
    # Линия выбранного порога
    fig_econ.add_vline(x=threshold, line_width=2, line_color="black")
    # Линия математического оптимума (из ВКР)
    fig_econ.add_vline(x=0.912, line_width=2, line_dash="dash", line_color="blue", annotation_text="Оптимум по F1 (0.912)")
    # Линия БИЗНЕС оптимума
    fig_econ.add_vline(x=best_biz_thr, line_width=2, line_dash="dash", line_color="green", annotation_text=f"Бизнес-оптимум ({best_biz_thr:.2f})")
    
    st.plotly_chart(fig_econ, use_container_width=True)

# ==========================================
# ВКЛАДКА 2: СИМУЛЯТОР (WHAT-IF)
# ==========================================
with tab2:
    st.markdown("### Интерактивный симулятор транзакций")
    st.markdown("Изменяйте главные признаки транзакции (выявленные через SHAP) и смотрите, как модель реагирует в реальном времени.")
    
    col_setup, col_viz = st.columns([1, 2])
    
    with col_setup:
        tx_type = st.radio("Взять за основу:", ["Легитимная операция", "Мошенничество"], horizontal=True)
        
        if 'session_idx' not in st.session_state:
            st.session_state.session_idx = df_processed[df_processed['Class'] == 0].sample(1).index[0]
            
        if st.button("Случайная транзакция"):
            class_label = 0 if tx_type == "Легитимная операция" else 1
            st.session_state.session_idx = df_processed[df_processed['Class'] == class_label].sample(1).index[0]
            
        idx = st.session_state.session_idx
        current_tx = df_processed.drop('Class', axis=1).loc[[idx]].copy()
        
        orig_amt = df_orig.loc[idx, 'Amount']
        orig_time = df_orig.loc[idx, 'Time']
        st.markdown(f"**Оригинальная сумма:** ${orig_amt:,.2f} | **Время:** {orig_time} сек.")
        
        st.markdown("#### Ручное изменение топ-признаков:")
        st.caption("Поведенческие PCA-признаки, которые реально влияют на модель.")
        
        new_v14 = st.slider("Признак V14 (Топ-1 триггер)", min_value=-20.0, max_value=15.0, value=float(current_tx['V14'].values[0]), step=0.5)
        current_tx['V14'] = new_v14
        
        new_v4 = st.slider("Признак V4 (Топ-2)", min_value=-10.0, max_value=15.0, value=float(current_tx['V4'].values[0]), step=0.5)
        current_tx['V4'] = new_v4
        
        new_v12 = st.slider("Признак V12 (Топ-3)", min_value=-20.0, max_value=15.0, value=float(current_tx['V12'].values[0]), step=0.5)
        current_tx['V12'] = new_v12

    with col_viz:
        prob = model.predict_proba(current_tx)[0, 1]
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'valueformat': ".2f"},
            title={'text': "Вероятность фрода", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob >= threshold else "green"},
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "salmon"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=40, r=40, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("**Объяснение решения (SHAP):** Почему модель выдала такой Risk Score?")
        explainer = shap.Explainer(model)
        shap_values = explainer(current_tx)
        
        fig_shap, ax_shap = plt.subplots(figsize=(8, 3))
        shap.plots.waterfall(shap_values[0], show=False, max_display=5)
        st.pyplot(fig_shap)

# ==========================================
# ВКЛАДКА 3: АНАЛИТИКА (EDA)
# ==========================================
with tab3:
    st.markdown("### Разведочный анализ (EDA)")
    c1, c2 = st.columns(2)
    
    df_plot = df_orig.copy()
    df_plot['Class_Name'] = df_plot['Class'].map({0: 'Легитимные', 1: 'Фрод'})
    
    with c1:
        class_counts = df_plot['Class_Name'].value_counts()
        fig_pie = px.pie(
            values=class_counts.values, 
            names=class_counts.index, 
            hole=0.5, 
            color_discrete_sequence=['#636EFA', '#EF553B'],
            title="Дисбаланс классов (тестовая выборка)"
        )
        fig_pie.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        fig_box = px.box(
            df_plot, 
            x="Class_Name", 
            y="Amount", 
            color="Class_Name", 
            log_y=True, 
            color_discrete_sequence=['#636EFA', '#EF553B'],
            title="Распределение сумм (Логарифмическая шкала)"
        )
        fig_box.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)