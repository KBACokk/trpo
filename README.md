# Тестирование CBLAS 

Этот проект содержит небольшие тесты для проверки функциональности библиотеки CBLAS (Basic Linear Algebra Subprograms) с использованием функций из OpenBLAS.

## О библиотеке CBLAS

CBLAS — это C-интерфейс для библиотеки BLAS, которая предоставляет стандартизированные функции для выполнения основных операций линейной алгебры. В данном проекте тестируются операции уровня 2:

- **GEMV** - умножение матрицы на вектор (general matrix-vector multiply)
- **TRMV** - умножение треугольной матрицы на вектор (triangular matrix-vector multiply)
- **SYMV** - умножение симметричной матрицы на вектор (symmetric matrix-vector multiply)
- **GER** - внешнее произведение векторов (rank-1 update)
- **TRSV** - решение треугольной системы (triangular solve)

## Запуск тестов

```bash
/mnt/c/Users/alt_ma1n$ make 
/mnt/c/Users/alt_ma1n$ python cblas_test.py
```
