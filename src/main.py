"""
Основной модуль Educational Plagiarism Detector.
"""

import os
import json
import argparse
import datetime
import glob

from .utils import (
    preprocess_text,
    tokenize_and_lemmatize,
    create_similarity_matrix,
    read_text_file,
    extract_text_from_pdf
)


def detect_plagiarism_in_directory(directory: str) -> dict:
    """
    Обнаружение плагиата во всех файлах указанной директории.
    
    Args:
        directory: Путь к директории с файлами
        
    Returns:
        Словарь с результатами анализа
    """
    print(f"🔍 Анализ файлов в директории: {directory}")
    
    # Получение списка файлов
    supported_extensions = ['.txt', '.pdf', '.docx']
    files = []
    
    for ext in supported_extensions:
        pattern = os.path.join(directory, f'*{ext}')
        files.extend(glob.glob(pattern))
    
    if len(files) < 2:
        return {
            'status': 'error',
            'message': f'Need at least 2 files for comparison. Found: {len(files)}',
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    print(f"📁 Найдено файлов: {len(files)}")
    
    # Чтение и обработка файлов
    texts = []
    filenames = []
    errors = []
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"  📄 Обработка: {filename}")
        
        try:
            # Определение типа файла и чтение
            if filepath.endswith('.txt'):
                text = read_text_file(filepath)
            elif filepath.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                errors.append(f"Unsupported file type: {filename}")
                continue
            
            if not text.strip():
                errors.append(f"Empty file: {filename}")
                continue
            
            # Предварительная обработка
            cleaned_text = preprocess_text(text)
            tokens = tokenize_and_lemmatize(cleaned_text)
            processed_text = ' '.join(tokens)
            
            texts.append(processed_text)
            filenames.append(filename)
            
            print(f"    ✅ Успешно ({len(tokens)} токенов)")
            
        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            errors.append(error_msg)
            print(f"    ❌ {error_msg}")
    
    if len(texts) < 2:
        return {
            'status': 'error',
            'message': 'Less than 2 valid files after processing',
            'errors': errors,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    # Вычисление схожести
    print("\n🔬 Вычисление схожести...")
    similarity_results = create_similarity_matrix(texts)
    
    # Подготовка результатов
    result = {
        'status': 'success',
        'timestamp': datetime.datetime.now().isoformat(),
        'total_files': len(files),
        'processed_files': len(texts),
        'filenames': filenames,
        'errors': errors,
        'similarity_matrices': {
            method: matrix.tolist()
            for method, matrix in similarity_results.items()
        }
    }
    
    # Сохранение результатов
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(results_dir, f'plagiarism_results_{timestamp}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены: {output_file}")
    
    # Вывод сводки
    print("\n📊 СВОДКА РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    
    combined_matrix = similarity_results['combined']
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            similarity = combined_matrix[i][j]
            
            if similarity > 0.7:
                status = "🚨 ВЫСОКАЯ ВЕРОЯТНОСТЬ ПЛАГИАТА"
                emoji = "⚠️"
            elif similarity > 0.4:
                status = "⚠️  УМЕРЕННАЯ СХОЖЕСТЬ"
                emoji = "🔍"
            else:
                status = "✅ НИЗКАЯ СХОЖЕСТЬ"
                emoji = "✓"
            
            print(f"{emoji} {filenames[i]} vs {filenames[j]}: {similarity:.2%} - {status}")
    
    print("=" * 60)
    
    return result


def analyze_single_pair(file1: str, file2: str) -> dict:
    """
    Анализ схожести между двумя файлами.
    
    Args:
        file1: Путь к первому файлу
        file2: Путь ко второму файлу
        
    Returns:
        Словарь с результатами сравнения
    """
    print("🔍 Сравнение файлов:")
    print(f"   1. {os.path.basename(file1)}")
    print(f"   2. {os.path.basename(file2)}")
    
    # Чтение файлов
    texts = []
    filenames = []
    
    for filepath in [file1, file2]:
        filename = os.path.basename(filepath)
        
        try:
            if filepath.endswith('.txt'):
                text = read_text_file(filepath)
            elif filepath.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
            
            # Обработка текста
            cleaned_text = preprocess_text(text)
            tokens = tokenize_and_lemmatize(cleaned_text)
            processed_text = ' '.join(tokens)
            
            texts.append(processed_text)
            filenames.append(filename)
            
            print(f"   ✅ {filename}: {len(tokens)} токенов")
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error processing {filename}: {str(e)}',
                'timestamp': datetime.datetime.now().isoformat()
            }
    
    # Вычисление схожести
    from .utils import (
        calculate_cosine_similarity,
        calculate_lcs_similarity,
        calculate_ngram_similarity
    )
    
    # Косинусная схожесть
    cosine_sim = calculate_cosine_similarity(texts)[0][1]
    
    # LCS схожесть
    lcs_sim = calculate_lcs_similarity(texts[0], texts[1])
    
    # N-gram схожесть
    ngram_sim = calculate_ngram_similarity(texts[0], texts[1])
    
    # Комбинированная схожесть
    combined_sim = 0.5 * cosine_sim + 0.3 * lcs_sim + 0.2 * ngram_sim
    
    result = {
        'status': 'success',
        'timestamp': datetime.datetime.now().isoformat(),
        'files': filenames,
        'similarity_scores': {
            'cosine': cosine_sim,
            'lcs': lcs_sim,
            'ngram': ngram_sim,
            'combined': combined_sim
        },
        'interpretation': interpret_similarity(combined_sim)
    }
    
    print("\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Cosine Similarity: {cosine_sim:.2%}")
    print(f"   LCS Similarity: {lcs_sim:.2%}")
    print(f"   N-gram Similarity: {ngram_sim:.2%}")
    print(f"   Combined Similarity: {combined_sim:.2%}")
    print(f"   Вердикт: {result['interpretation']}")
    
    return result


def interpret_similarity(score: float) -> str:
    """
    Интерпретация коэффициента схожести.
    
    Args:
        score: Коэффициент схожести (0-1)
        
    Returns:
        Текстовая интерпретация
    """
    if score > 0.7:
        return "🚨 ВЫСОКАЯ ВЕРОЯТНОСТЬ ПЛАГИАТА - требуется проверка преподавателя"
    elif score > 0.4:
        return "⚠️  УМЕРЕННАЯ СХОЖЕСТЬ - возможны заимствования"
    elif score > 0.2:
        return "🔍 НЕБОЛЬШАЯ СХОЖЕСТЬ - вероятно случайные совпадения"
    else:
        return "✅ НИЗКАЯ СХОЖЕСТЬ - плагиат маловероятен"


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Educational Plagiarism Detector - система обнаружения плагиата в студенческих работах'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Команда для анализа директории
    dir_parser = subparsers.add_parser('analyze-dir', help='Analyze all files in a directory')
    dir_parser.add_argument('directory', help='Directory with student files')
    
    # Команда для сравнения двух файлов
    pair_parser = subparsers.add_parser('compare', help='Compare two specific files')
    pair_parser.add_argument('file1', help='First file')
    pair_parser.add_argument('file2', help='Second file')
    
    args = parser.parse_args()
    
    if args.command == 'analyze-dir':
        detect_plagiarism_in_directory(args.directory)
    elif args.command == 'compare':
        analyze_single_pair(args.file1, args.file2)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()