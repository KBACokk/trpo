import ctypes
import numpy as np

CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasUpper = 121
CblasLower = 122
CblasNonUnit = 131
CblasUnit = 132

lib = ctypes.CDLL("./libopenblas_zenp-r0.3.31.dev.so")

lib.cblas_sgemv.argtypes = [
    ctypes.c_int, ctypes.c_int,  
    ctypes.c_int, ctypes.c_int,  
    ctypes.c_float,              
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,  
    ctypes.c_float,              
    ctypes.POINTER(ctypes.c_float), ctypes.c_int   
]

lib.cblas_dgemv.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

lib.cblas_strmv.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.cblas_ssymv.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.cblas_sger.argtypes = [
    ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.cblas_strsv.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]

lib.cblas_sgbmv.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int
]


def set_threads(n):
    lib.openblas_set_num_threads(ctypes.c_int(n))
    print(f"  Потоков: {n}")

def test_gemv():
    print("\n--- GEMV ---")
    m, n = 3, 2
    
    for threads in [1, 4, 8]:
        set_threads(threads)
        
        # float
        A = np.array([[1,2],[3,4],[5,6]], dtype=np.float32)
        x = np.array([1,2], dtype=np.float32)
        y = np.zeros(m, dtype=np.float32)
        
        lib.cblas_sgemv(
            CblasRowMajor, CblasNoTrans, m, n,
            2.0, 
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
            1.0,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1
        )
        
        expected = 2.0 * A @ x
        if np.allclose(y, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")
        
        A = np.array([[1,2],[3,4],[5,6]], dtype=np.float64)
        x = np.array([1,2], dtype=np.float64)
        y = np.zeros(m, dtype=np.float64)
        
        lib.cblas_dgemv(
            CblasRowMajor, CblasNoTrans, m, n,
            2.0, 
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
            1.0,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1
        )
        
        expected = 2.0 * A @ x
        if np.allclose(y, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")

def test_trmv():
    print("\n--- TRMV ---")
    n = 3
    
    for threads in [1, 4, 8]:
        set_threads(threads)
        
        A = np.array([[1,2,3],[0,4,5],[0,0,6]], dtype=np.float32)
        x = np.array([1,2,3], dtype=np.float32)
        x_orig = x.copy()
        
        lib.cblas_strmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
            n,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1
        )
        
        expected = A @ x_orig
        if np.allclose(x, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")

def test_symv():
    print("\n--- SYMV ---")
    n = 3
    
    for threads in [1, 4, 8]:
        set_threads(threads)
        
        A = np.array([[1,2,3],[2,4,5],[3,5,6]], dtype=np.float32)
        x = np.array([1,2,3], dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        
        lib.cblas_ssymv(
            CblasRowMajor, CblasUpper, n,
            2.0,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
            1.0,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1
        )
        
        expected = 2.0 * A @ x
        if np.allclose(y, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")


def test_ger():
    print("\n--- GER ---")
    m, n = 3, 2
    
    for threads in [1, 4, 8]:
        set_threads(threads)
        
        A = np.ones((m, n), dtype=np.float32) * 2
        x = np.array([1,2,3], dtype=np.float32)
        y = np.array([4,5], dtype=np.float32)
        
        lib.cblas_sger(
            CblasRowMajor, m, n,
            2.0,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n
        )
        
        expected = np.ones((m, n)) * 2 + 2.0 * np.outer(x, y)
        if np.allclose(A, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")


def test_trsv():
    print("\n--- TRSV ---")
    n = 3
    
    for threads in [1, 4, 8]:
        set_threads(threads)
        
        A = np.array([[2,1,1],[0,2,1],[0,0,2]], dtype=np.float32)
        b = np.array([4,5,6], dtype=np.float32)
        b_orig = b.copy()
        
        lib.cblas_strsv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
            n,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1
        )
        
        expected = np.linalg.solve(A, b_orig)
        if np.allclose(b, expected):
            print(f"  Проход: Успешно")
        else:
            print(f"  Проход: Провал")


if __name__ == "__main__":
    print("="*50)
    print("="*50)
    
    test_gemv()
    test_trmv()
    test_symv()
    test_ger()
    test_trsv()
    
    print("\n" + "="*50)
    print("Тесты пройдены")
