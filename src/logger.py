import logging
import sys


def get_logger(name: str = "FraudDetection") -> logging.Logger:
    """
    Создает и настраивает логгер для проекта.
    """
    logger = logging.getLogger(name)
    
    # Защита от дублирования логов при перезапуске ячеек в Jupyter
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Настраиваем вывод в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Формат логов: Время - Имя - Уровень - Сообщение
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger