# Используем легковесный образ Python
FROM python:3.10-slim

# Скачиваем бинарник пакетного менеджера uv (это ускорит сборку в разы)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Задаем рабочую директорию
WORKDIR /app

# Копируем конфигурацию проекта
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости системы и питона (без создания виртуального окружения внутри докера)
RUN uv sync --frozen --no-dev

# Копируем весь оставшийся код и данные
COPY src/ /app/src/
COPY data/ /app/data/
COPY xgb_model.json /app/

# Открываем порт для Streamlit
EXPOSE 8501

# Команда для запуска веб-приложения через окружение uv
CMD ["uv", "run", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]