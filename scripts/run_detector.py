#!/usr/bin/env python3
"""
Скрипт для запуска плагиат-детектора.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import detect_plagiarism_in_directory

if __name__ == '__main__':
    # По умолчанию анализируем папку uploads
    uploads_dir = 'uploads'
    
    if not os.path.exists(uploads_dir):
        print(f"Создаю папку {uploads_dir}...")
        os.makedirs(uploads_dir)
        print(f"Поместите файлы студентов в папку {uploads_dir} и запустите снова.")
        sys.exit(0)
    
    detect_plagiarism_in_directory(uploads_dir)