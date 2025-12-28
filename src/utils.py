"""
Утилиты для обработки текста и вычисления схожести.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Скачиваем ресурсы NLTK при первом импорте
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def preprocess_text(text: str, language: str = 'english') -> str:
    """
    Предварительная обработка текста.
    
    Args:
        text: Исходный текст
        language: Язык текста ('english' или 'russian')
        
    Returns:
        Обработанный текст
    """
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление пунктуации и специальных символов
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    
    # Удаление лишних пробелов
    text = ' '.join(text.split())
    
    return text


def tokenize_and_lemmatize(text: str, language: str = 'english') -> list:
    """
    Токенизация и лемматизация текста.
    
    Args:
        text: Очищенный текст
        language: Язык текста
        
    Returns:
        Список лемматизированных токенов
    """
    # Токенизация
    tokens = word_tokenize(text, language=language)
    
    # Удаление стоп-слов
    stop_words = set(stopwords.words(language))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def calculate_cosine_similarity(texts: list) -> np.ndarray:
    """
    Вычисление косинусной схожести между текстами.
    
    Args:
        texts: Список текстов
        
    Returns:
        Матрица схожести N x N
    """
    # Создание TF-IDF векторов
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Вычисление косинусной схожести
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix


def calculate_lcs_similarity(text1: str, text2: str) -> float:
    """
    Вычисление схожести на основе Longest Common Subsequence.
    
    Args:
        text1: Первый текст
        text2: Второй текст
        
    Returns:
        Коэффициент схожести от 0 до 1
    """
    # Токенизация
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    m, n = len(tokens1), len(tokens2)
    
    # Динамическое программирование для LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    max_length = max(m, n)
    
    return lcs_length / max_length if max_length > 0 else 0.0


def calculate_ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Вычисление схожести на основе n-грамм.
    
    Args:
        text1: Первый текст
        text2: Второй текст
        n: Размер n-грамм
        
    Returns:
        Коэффициент Жаккара для n-грамм
    """
    # Генерация n-грамм
    def get_ngrams(text: str, n: int) -> set:
        tokens = text.split()
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    # Коэффициент Жаккара
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    
    return intersection / union if union > 0 else 0.0


def read_text_file(filepath: str) -> str:
    """
    Чтение текстового файла с автоматическим определением кодировки.
    
    Args:
        filepath: Путь к файлу
        
    Returns:
        Содержимое файла как строка
    """
    encodings = ['utf-8', 'cp1251', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Cannot read file {filepath} with any encoding")


def extract_text_from_pdf(filepath: str) -> str:
    """
    Извлечение текста из PDF файла.
    
    Args:
        filepath: Путь к PDF файлу
        
    Returns:
        Текст из PDF
    """
    text = ""
    
    # Попробуем PyPDF2
    try:
        import PyPDF2
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except ImportError:
        pass
    
    # Если PyPDF2 не сработал или не установлен, пробуем pdfplumber
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except ImportError:
            raise ImportError("Install PyPDF2 or pdfplumber to read PDF files")
    
    return text


def create_similarity_matrix(texts: list) -> dict:
    """
    Создание матриц схожести всеми методами.
    
    Args:
        texts: Список текстов
        
    Returns:
        Словарь с матрицами схожести для каждого метода
    """
    n = len(texts)
    
    # Косинусная схожесть
    cosine_matrix = calculate_cosine_similarity(texts)
    
    # Матрицы для LCS и n-gram
    lcs_matrix = np.zeros((n, n))
    ngram_matrix = np.zeros((n, n))
    
    # Попарное сравнение
    for i in range(n):
        for j in range(i, n):
            if i == j:
                lcs_matrix[i][j] = 1.0
                ngram_matrix[i][j] = 1.0
            else:
                lcs_matrix[i][j] = calculate_lcs_similarity(texts[i], texts[j])
                lcs_matrix[j][i] = lcs_matrix[i][j]
                
                ngram_matrix[i][j] = calculate_ngram_similarity(texts[i], texts[j])
                ngram_matrix[j][i] = ngram_matrix[i][j]
    
    # Комбинированная схожесть (взвешенное среднее)
    combined_matrix = (
        0.5 * cosine_matrix +
        0.3 * lcs_matrix +
        0.2 * ngram_matrix
    )
    
    return {
        'cosine': cosine_matrix,
        'lcs': lcs_matrix,
        'ngram': ngram_matrix,
        'combined': combined_matrix
    }