"""
Тесты для основного модуля плагиат-детектора.
"""

import os
import tempfile
import pytest
from src.main import (
    detect_plagiarism_in_directory,
    analyze_single_pair,
    interpret_similarity,
)


class TestPlagiarismDetector:
    """Тесты для функций обнаружения плагиата."""

    def test_interpret_similarity(self):
        """Тест интерпретации коэффициента схожести."""
        assert "🚨 ВЫСОКАЯ" in interpret_similarity(0.8)
        assert "⚠️" in interpret_similarity(0.5)
        assert "🔍" in interpret_similarity(0.3)
        assert "✅" in interpret_similarity(0.1)

    def test_analyze_single_pair_text_files(self):
        """Тест сравнения двух текстовых файлов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Создаём два файла
            file1 = os.path.join(tmpdir, "essay1.txt")
            file2 = os.path.join(tmpdir, "essay2.txt")

            # Похожий контент
            with open(file1, "w", encoding="utf-8") as f:
                f.write("Artificial intelligence is important for education.")

            with open(file2, "w", encoding="utf-8") as f:
                f.write("Education benefits from artificial intelligence.")

            result = analyze_single_pair(file1, file2)

            assert result["status"] == "success"
            assert len(result["files"]) == 2
            assert "similarity_scores" in result
            
            scores = result["similarity_scores"].values()
            assert all(0 <= score <= 1 for score in scores)

    def test_detect_plagiarism_in_directory_empty(self):
        """Тест анализа пустой директории."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_plagiarism_in_directory(tmpdir)

            assert result["status"] == "error"
            assert "Need at least 2 files" in result["message"]


class TestIntegration:
    """Интеграционные тесты."""

    def test_full_pipeline(self):
        """Тест полного пайплайна с созданием файлов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Создаём тестовые файлы
            files_data = [
                ("essay1.txt", "Machine learning algorithms can improve student performance."),
                ("essay2.txt", "Algorithms for machine learning help improve performance of students."),
                ("essay3.txt", "Climate change is a major global environmental issue."),
            ]

            for filename, content in files_data:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            # Запускаем анализ
            result = detect_plagiarism_in_directory(tmpdir)

            # Проверяем результаты
            assert result["status"] == "success"
            assert result["processed_files"] == 3
            assert len(result["filenames"]) == 3
            assert "similarity_matrices" in result

            # Проверяем, что матрицы имеют правильный размер
            for method, matrix in result["similarity_matrices"].items():
                assert len(matrix) == 3  # 3 файла
                assert len(matrix[0]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])