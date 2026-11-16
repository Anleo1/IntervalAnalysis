# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 по интервальному анализу.
Особенность интервальных матриц A1 (3x2: томография / регрессия) и A2 (3x3).

Для каждой матрицы:
  1) находится минимальное delta*, при котором матрица содержит особенную точечную матрицу;
  2) строится такая точечная матрица A';
  3) проверяются принадлежность A' интервальной матрице и её особенность.
"""
import itertools
import numpy as np
from scipy.optimize import minimize_scalar, brentq

np.set_printoptions(precision=6, suppress=True, linewidth=120)
TOL = 1e-12

class IntervalMatrix:
    """A = [mid - rad, mid + rad]."""

    def __init__(self, mid, rad):
        self.mid = np.asarray(mid, dtype=float)
        self.rad = np.asarray(rad, dtype=float)

    @property
    def inf(self):
        return self.mid - self.rad

    @property
    def sup(self):
        return self.mid + self.rad

    def contains(self, M, tol=TOL):
        """A' принадлежит A  <=>  inf A <= A' <= sup A (поэлементно)."""
        M = np.asarray(M, dtype=float)
        return bool(np.all(M >= self.inf - tol) and np.all(M <= self.sup + tol))

    def vertices(self):
        """Перебор всех вершин: каждый элемент равен mid -+ rad (2^k матриц)."""
        idx = np.where(self.rad.ravel() > 0)[0]
        for signs in itertools.product((-1.0, 1.0), repeat=len(idx)):
            v = self.mid.ravel().copy()
            v[idx] += np.array(signs) * self.rad.ravel()[idx]
            yield v.reshape(self.mid.shape)


def print_intervals(A, title):
    print(f"\n{title}")
    for i in range(A.mid.shape[0]):
        print("   " + "  ".join(f"[{A.inf[i, j]:.6f}; {A.sup[i, j]:.6f}]" for j in range(A.mid.shape[1])))

MID_A1 = np.array([[0.95, 1.00],
                   [1.05, 1.00],
                   [1.10, 1.00]])
R_TOMO = np.ones((3, 2))                            # rad A1 = delta * R_TOMO
R_REGR = np.array([[1., 0.], [1., 0.], [1., 0.]])   # rad A1 = delta * R_REGR


def delta_of_x(x, mid, R):
    """
    Наименьшее delta, при котором в [mid - delta*R, mid + delta*R] есть матрица A' с A'x = 0.
    Критерий Оеттли-Прагера:  |mid x| <= delta * R |x|  (поэлементно).
    """
    num = np.abs(mid @ x)
    den = R @ np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.where(num > 0, np.inf, 0.0))
    return float(r.max())


def min_delta_numeric(mid, R):
    """Численная проверка: минимум delta(x) по направлениям x = (cos t, sin t), t in [0, pi)."""
    f = lambda t: delta_of_x(np.array([np.cos(t), np.sin(t)]), mid, R)
    ts = np.linspace(0.0, np.pi, 4001)
    k = int(np.argmin([f(t) for t in ts]))
    res = minimize_scalar(f, bounds=(ts[max(k - 1, 0)], ts[min(k + 1, 4000)]),
                          method="bounded", options={"xatol": 1e-14})
    return res.fun


def analyze_A1(case):
    m = MID_A1[:, 0]
    if case == "регрессия":
        R = R_REGR
        # второй столбец (1,1,1) точный => a1 = a2 = a3 = lam;
        # интервалы [m_i - d, m_i + d] имеют общую точку <=> d >= (max m - min m)/2
        delta = (m.max() - m.min()) / 2
        lam = m.max() - delta
        Ap = np.column_stack([np.full(3, lam), np.ones(3)])
    else:
        R = R_TOMO
        # a = lam * b, |m_i - lam| <= d (1 + lam); оптимальное lam - середина разброса m
        lam = (m.max() + m.min()) / 2
        delta = (m.max() - m.min()) / (2 * (1 + lam))
        b = np.array([1 - delta, 1.0, 1 + delta])      # b1, b3 - на границах, b2 - центр
        Ap = np.column_stack([lam * b, b])

    A = IntervalMatrix(MID_A1, delta * R)
    d_num = min_delta_numeric(MID_A1, R)

    print(f"A1, {case}")
    print(f"п.1  delta* (формула)   = {delta:.10f}   lambda = {lam:.6f}")
    print(f"     delta* (численно)  = {d_num:.10f}   (проверка через delta(x))")
    below = d_num <= 0.999 * delta
    above = d_num <= 1.001 * delta
    print(f"     при 0.999*delta*: особенная матрица есть? {below};  при 1.001*delta*: {above}")
    print(f"     диапазон: delta in [{delta:.6f}; +inf)")
    print_intervals(A, "A1(delta*):")
    print("\nп.2  A1' =\n", Ap)

    print("\nп.3  проверка")
    print("     A1' принадлежит A1:", A.contains(Ap))
    sv = np.linalg.svd(Ap, compute_uv=False)
    print("     сингулярные числа A1':", sv, "-> rank =", np.linalg.matrix_rank(Ap, tol=1e-10))
    for i, j in itertools.combinations(range(3), 2):
        print(f"     минор строк ({i + 1},{j + 1}) = {np.linalg.det(Ap[[i, j], :]):.3e}")
    print(f"     det(A1'^T A1') = {np.linalg.det(Ap.T @ Ap):.3e}")
    x = np.array([1.0, -lam])
    print(f"     ядро: x = (1; -lambda) = {x}, A1' x = {Ap @ x}")
    return delta

MID_A2 = np.array([[1.10, 0.90, 1.10],
                   [1.40, 1.00, 0.80],
                   [0.80, 1.40, 1.20]])
E3 = np.ones((3, 3))


def analyze_A2():
    print("\n" + "=" * 74)
    print("A2 (3x3), rad A2 = delta * E")
    print("=" * 74)
    inv = np.linalg.inv(MID_A2)
    print(f"det(mid A2) = {np.linalg.det(MID_A2):.6f},  cond_2(mid A2) = {np.linalg.cond(MID_A2):.4f}")
    print("(mid A2)^(-1) =\n", inv)

    # --- способ 1: перебор 64 пар (y, z) из {-1, 1}^3 (формула Рона) ---
    best = (0.0, None, None)
    for y in itertools.product((-1.0, 1.0), repeat=3):
        for z in itertools.product((-1.0, 1.0), repeat=3):
            val = np.array(z) @ inv @ np.array(y)
            if abs(val) > abs(best[0]):
                best = (val, np.array(y), np.array(z))
    val, y, z = best
    mu = abs(val)
    delta_ron = 1.0 / mu
    print(f"\nспособ 1 (перебор 64 пар y, z):  max|z^T (mid A2)^-1 y| = {mu:.6f}"
          f"  при y = {y}, z = {z}")
    print(f"     delta* = 1/{mu:.6f} = {delta_ron:.10f}")

    # --- способ 2: перебор 512 вершин: det - мультилинейная функция => экстремумы в вершинах ---
    signs = [np.sign(V - MID_A2) for V in IntervalMatrix(MID_A2, E3).vertices()]

    def det_range(d):
        dets = [np.linalg.det(MID_A2 + d * s) for s in signs]
        return min(dets), max(dets)

    delta_vert = brentq(lambda d: det_range(d)[0], 0.0, 0.5, xtol=1e-15)
    print(f"способ 2 (перебор 2^9 = 512 вершин): min det обращается в 0 при delta = {delta_vert:.10f}")
    print("\n     delta     min det     max det")
    for d in (0.0, 0.05, 0.09, 0.10, 0.11, 0.15):
        lo, hi = det_range(d)
        print(f"     {d:5.2f}   {lo:9.5f}   {hi:9.5f}")
    print(f"\nп.1  диапазон: delta in [{delta_ron:.6f}; +inf)")

    # --- п.2: A2' = mid A2 - sign(val) * delta* * y z^T ---
    delta = delta_ron
    A = IntervalMatrix(MID_A2, delta * E3)
    print_intervals(A, "A2(delta*):")
    Ap = MID_A2 - np.sign(val) * delta * np.outer(y, z)
    print("\nп.2  A2' =\n", Ap)

    print("\nп.3  проверка")
    print("     A2' принадлежит A2:", A.contains(Ap))
    print("     |A2' - mid A2| =\n", np.abs(Ap - MID_A2))
    print(f"     det A2' = {np.linalg.det(Ap):.3e}")
    print("     сингулярные числа A2':", np.linalg.svd(Ap, compute_uv=False),
          "-> rank =", np.linalg.matrix_rank(Ap, tol=1e-10))
    x = inv @ y
    x = x / np.abs(x).max()
    print(f"     ядро: x = {x}, A2' x = {Ap @ x}")
    return delta


if __name__ == "__main__":
    d_tomo = analyze_A1("томография")
    print()
    d_regr = analyze_A1("регрессия")
    d_sq = analyze_A2()
    print("\n" + "=" * 74)
    print(f"ИТОГ: delta*(A1, томография) = {d_tomo:.8f} = 1/27")
    print(f"      delta*(A1, регрессия)  = {d_regr:.8f}")
    print(f"      delta*(A2)             = {d_sq:.8f}")
