# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 по интервальному анализу.
Особенность интервальных матриц A1 (3x2, томография / регрессия) и A2 (3x3).

Запуск:  python lab1_interval.py
Создаёт рисунки fig1.png ... fig4.png и печатает все числовые результаты.
Библиотеки: numpy, scipy, matplotlib.
"""
import itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize, minimize_scalar

np.set_printoptions(precision=6, suppress=True, linewidth=120)
TOL = 1e-12  # допуск при проверке принадлежности и особенности


# =====================================================================
# 0. Интервальная матрица и вспомогательные функции
# =====================================================================
class IntervalMatrix:
    """Интервальная матрица A = [mid - rad, mid + rad]."""

    def __init__(self, mid, rad):
        self.mid = np.asarray(mid, dtype=float)
        self.rad = np.asarray(rad, dtype=float)
        assert self.mid.shape == self.rad.shape and np.all(self.rad >= 0)

    @property
    def inf(self):
        return self.mid - self.rad

    @property
    def sup(self):
        return self.mid + self.rad

    def contains(self, M, tol=TOL):
        """Поэлементная проверка inf <= M <= sup (с допуском)."""
        M = np.asarray(M, dtype=float)
        return bool(np.all(M >= self.inf - tol) and np.all(M <= self.sup + tol))

    def vertices(self):
        """Все 2^k вершин (k - число элементов с ненулевым радиусом)."""
        flat_mid, flat_rad = self.mid.ravel(), self.rad.ravel()
        idx = np.where(flat_rad > 0)[0]
        for signs in itertools.product((-1.0, 1.0), repeat=len(idx)):
            v = flat_mid.copy()
            v[idx] += np.array(signs) * flat_rad[idx]
            yield v.reshape(self.mid.shape)


def show_interval_matrix(A, name):
    print(f"\n{name}: нижняя граница inf A и верхняя граница sup A")
    for i in range(A.mid.shape[0]):
        row = "  ".join(f"[{A.inf[i, j]:.6f}, {A.sup[i, j]:.6f}]" for j in range(A.mid.shape[1]))
        print("   ", row)


def sing_values(M):
    return np.linalg.svd(M, compute_uv=False)


def oettli_prager_delta(x, mid, rad1):
    """
    Критерий Оеттли-Прагера для однородной системы A x = 0:
        существует A из [mid - d*rad1, mid + d*rad1] с A x = 0
        <=>  |mid x| <= d * rad1 |x|   (покомпонентно).
    Для заданного x возвращает наименьшее такое d.
    """
    num = np.abs(mid @ x)
    den = rad1 @ np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.where(num > 0, np.inf, 0.0))
    return float(np.max(r))


def singular_matrix_from_x(x, mid, rad1, delta):
    """
    Строит точечную матрицу A' из интервальной матрицы [mid - delta*rad1, mid + delta*rad1],
    для которой A' x = 0. Для i-й строки: a_i = c_i - lam_i * sign(c_i x) * (rad_i * sign(x)),
    где lam_i = |c_i x| / (rad_i |x|) <= delta.
    """
    mid = np.asarray(mid, float)
    Ap = mid.copy()
    sx = np.sign(x)
    for i in range(mid.shape[0]):
        num = mid[i] @ x
        den = rad1[i] @ np.abs(x)
        if den > 0:
            lam = abs(num) / den
            Ap[i] = mid[i] - lam * np.sign(num) * rad1[i] * sx
    return Ap


# =====================================================================
# 1. Прямоугольная матрица A1 (3x2)
# =====================================================================
MID_A1 = np.array([[0.95, 1.00],
                   [1.05, 1.00],
                   [1.10, 1.00]])
RAD1_TOMO = np.ones((3, 2))                      # (3): rad A1 = delta * [[1,1],[1,1],[1,1]]
RAD1_REGR = np.array([[1., 0.], [1., 0.], [1., 0.]])  # (4): rad A1 = delta * [[1,0],[1,0],[1,0]]


def min_delta_rect_numeric(mid, rad1, n_grid=200001):
    """
    Численный (независимый) поиск min delta: минимизируем delta(x) из критерия
    Оеттли-Прагера по направлению x = (cos t, sin t), t из [0, pi).
    Сначала грубый перебор по сетке, затем уточнение методом Брента.
    """
    f = lambda t: oettli_prager_delta(np.array([np.cos(t), np.sin(t)]), mid, rad1)
    ts = np.linspace(0, np.pi, n_grid)
    vals = np.array([f(t) for t in ts[::50]])           # грубая сетка (каждая 50-я точка)
    k = int(np.argmin(vals)) * 50
    lo, hi = ts[max(k - 50, 0)], ts[min(k + 50, n_grid - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-15})
    x = np.array([np.cos(res.x), np.sin(res.x)])
    return res.fun, x


def analyze_rectangular(label, rad1):
    print("=" * 78)
    print(f"A1, случай: {label}")
    print("=" * 78)

    # ---------- Пункт 1: минимальное delta -----------------------------
    if label == "регрессия":
        # Второй столбец точечный (1,1,1). Особенность <=> первый столбец
        # коллинеарен (1,1,1) <=> у трёх интервалов [c_i - d, c_i + d] есть общая точка
        # <=> max(c_i - d) <= min(c_i + d) <=> d >= (max c - min c)/2
        c = MID_A1[:, 0]
        d_an = (c.max() - c.min()) / 2
        print(f"Аналитически: delta* = (max c - min c)/2 = ({c.max()} - {c.min()})/2 = {d_an:.10f}")
    else:
        # Столбцы u = t*v (rank 1, t>0). Из условия пересечения интервалов:
        #   t(1+d) >= c_max - d  и  t(1-d) <= c_min + d
        #   => (c_max - d)/(1+d) <= t <= (c_min + d)/(1-d)
        #   нужно (c_max - d)(1 - d) <= (c_min + d)(1 + d)
        cmax, cmin = MID_A1[:, 0].max(), MID_A1[:, 0].min()
        # (cmax - d)(1 - d) <= (cmin + d)(1 + d)
        # cmax - (1+cmax) d + d^2 <= cmin + (1+cmin) d + d^2
        # d >= (cmax - cmin) / (2 + cmax + cmin)
        d_an = (cmax - cmin) / (2 + cmax + cmin)
        print(f"Аналитически: delta* = (c_max - c_min)/(2 + c_max + c_min) = "
              f"{cmax - cmin:.2f}/{2 + cmax + cmin:.2f} = {d_an:.10f}  (= 1/27)")

    d_num, x_num = min_delta_rect_numeric(MID_A1, rad1)
    print(f"Численно (Оеттли-Прагер + оптимизация по направлению x): delta* = {d_num:.10f}")
    print(f"   оптимальное направление x* = {x_num} (с точностью до масштаба)")
    print(f"Диапазон: матрица A1 содержит матрицу неполного столбцового ранга при delta >= {d_an:.10f}")

    # Проверка «до» и «после» порога: min delta(x) и ранги/сингулярные числа
    for d in (0.5 * d_an, 0.999 * d_an, d_an, 1.001 * d_an, 2 * d_an):
        feasible = d_num <= d + 1e-6   # допуск на погрешность численной минимизации
        print(f"   delta = {d:.8f}: особенная матрица в A1 {'ЕСТЬ' if feasible else 'нет'}")

    # ---------- Пункт 2: построение A1' --------------------------------
    delta = d_an
    A = IntervalMatrix(MID_A1, delta * rad1)
    show_interval_matrix(A, f"A1(delta*) [{label}]")

    if label == "регрессия":
        t = 0.5 * (MID_A1[:, 0].max() + MID_A1[:, 0].min())  # общая точка = 1.025
        Ap = np.column_stack([np.full(3, t), np.ones(3)])
    else:
        t = (MID_A1[:, 0].max() - delta) / (1 + delta)       # = 1.025
        v = np.array([1 - delta, 1.0, 1 + delta])            # второй столбец A1'
        u = t * v                                            # первый столбец = t * второй
        Ap = np.column_stack([u, v])
    print("\nПостроенная матрица A1' (аналитически):\n", Ap)

    # Эта же матрица, построенная по численному x*:
    Ap_num = singular_matrix_from_x(x_num, MID_A1, rad1, delta)
    print("Матрица, построенная численно по x* (для сверки):\n", Ap_num)

    # ---------- Пункт 3: проверка принадлежности и особенности ----------
    print("\nПроверка:")
    print("  A1' in A1 :", A.contains(Ap))
    print("  A1' - inf A1 >= 0 :", np.all(Ap - A.inf >= -TOL), "; sup A1 - A1' >= 0 :", np.all(A.sup - Ap >= -TOL))
    sv = sing_values(Ap)
    print(f"  сингулярные числа A1': {sv}  -> rank = {np.linalg.matrix_rank(Ap, tol=1e-10)} (< 2)")
    print("  все миноры 2x2:")
    minors = []
    for i, j in itertools.combinations(range(3), 2):
        m = np.linalg.det(Ap[[i, j], :])
        minors.append(m)
        print(f"     строки ({i + 1},{j + 1}): det = {m:.3e}")
    print("  det(A1'^T A1') =", f"{np.linalg.det(Ap.T @ Ap):.3e}")
    ker = np.linalg.svd(Ap)[2][-1]
    print(f"  ненулевой x: A1' x = 0 при x = {ker}, ||A1' x|| = {np.linalg.norm(Ap @ ker):.3e}")
    # для сравнения - при меньшем delta такой матрицы нет
    d_small = 0.9 * delta
    As = IntervalMatrix(MID_A1, d_small * rad1)
    print(f"  для сравнения: при delta = 0.9 delta* = {d_small:.6f} оптимум delta(x) = {d_num:.6f} > delta ->"
          f" особенных матриц в A1 нет; sigma_min(mid A1) = {sing_values(MID_A1)[-1]:.6f}")
    return dict(delta=delta, Ap=Ap, A=A, t=t, x=x_num, d_num=d_num)


# =====================================================================
# 2. Квадратная матрица A2 (3x3)
# =====================================================================
MID_A2 = np.array([[1.10, 0.90, 1.10],
                   [1.40, 1.00, 0.80],
                   [0.80, 1.40, 1.20]])
RAD2_1 = np.ones((3, 3))


def analyze_square():
    print("\n" + "=" * 78)
    print("A2 (3x3), rad A2 = delta * E")
    print("=" * 78)
    n = 3
    print("det(mid A2) =", f"{np.linalg.det(MID_A2):.6f}", " cond_2(mid A2) =", f"{np.linalg.cond(MID_A2):.4f}")
    Ainv = np.linalg.inv(MID_A2)
    print("(mid A2)^(-1) =\n", Ainv)

    # ---------- Пункт 1: радиус неособенности (формула Рона) ------------
    # delta* = 1 / max_{y,z in {±1}^n} rho_0( (mid A)^{-1} D_y E D_z ),
    # rho_0 - максимальное по модулю ВЕЩЕСТВЕННОЕ собственное значение.
    # Для E = e e^T имеем D_y E D_z = y z^T (ранг 1), тогда
    # (mid A)^{-1} y z^T имеет единственное ненулевое с.з. z^T (mid A)^{-1} y.
    best, best_y, best_z = -np.inf, None, None
    table = []
    for y in itertools.product((-1.0, 1.0), repeat=n):
        for z in itertools.product((-1.0, 1.0), repeat=n):
            y_, z_ = np.array(y), np.array(z)
            M = Ainv @ np.outer(y_, z_)
            ev = np.linalg.eigvals(M)
            real_ev = ev[np.abs(ev.imag) < 1e-10].real
            rho0 = np.max(np.abs(real_ev)) if real_ev.size else 0.0
            lam = z_ @ Ainv @ y_                 # то же значение в явном виде
            table.append((y, z, lam, rho0))
            if lam > best:
                best, best_y, best_z = lam, y_, z_
    mu = best
    delta = 1.0 / mu
    print(f"\nmax_{{y,z}} z^T (mid A)^(-1) y = {mu:.10f}  при y = {best_y}, z = {best_z}")
    print(f"=> delta* = 1/{mu:.6f} = {delta:.10f}")
    rho_check = max(t[3] for t in table)
    print(f"   проверка через собственные числа: max rho_0 = {rho_check:.10f} (совпадает: {np.isclose(rho_check, mu)})")

    # Независимая численная проверка: минимизация delta(x) по x на сфере (Оеттли-Прагер)
    f = lambda w: oettli_prager_delta(w / np.linalg.norm(w), MID_A2, RAD2_1)
    rng = np.random.default_rng(0)
    best_val, best_x = np.inf, None
    for _ in range(60):
        w0 = rng.normal(size=3)
        r = minimize(f, w0, method="Nelder-Mead", options=dict(xatol=1e-13, fatol=1e-15, maxiter=20000))
        if r.fun < best_val:
            best_val, best_x = r.fun, r.x / np.linalg.norm(r.x)
    print(f"Численная проверка (Оеттли-Прагер, Nelder-Mead, 60 стартов): delta* ~ {best_val:.8f}")
    print(f"   x* ~ {best_x}   (аналитический x = (mid A)^(-1) y = {Ainv @ best_y / np.linalg.norm(Ainv @ best_y)})")

    # Дополнительный анализ: крайние значения det по вершинам при разных delta
    ds = np.linspace(0, 0.2, 401)
    dmin, dmax = [], []
    verts0 = [v for v in IntervalMatrix(MID_A2, RAD2_1).vertices()]  # вершины при delta=1 (используем знаки)
    signs = [np.sign(v - MID_A2) for v in verts0]
    for d in ds:
        dets = [np.linalg.det(MID_A2 + d * s) for s in signs]
        dmin.append(min(dets))
        dmax.append(max(dets))
    dmin, dmax = np.array(dmin), np.array(dmax)
    # delta, при котором min det впервые достигает 0 (поиск по сетке + бисекция)
    from scipy.optimize import brentq
    g = lambda d: min(np.linalg.det(MID_A2 + d * s) for s in signs)
    sign0 = np.sign(g(0))
    d_vert = brentq(lambda d: g(d) if sign0 > 0 else -g(d), 0.0, 0.2, xtol=1e-15)
    print(f"Проверка перебором 2^9 = 512 вершин: det-диапазон обращается в 0 при delta = {d_vert:.10f}")
    print(f"   det(mid A2) = {np.linalg.det(MID_A2):.6f}; при delta=0.9*delta*: min det = {g(0.9 * delta):.6f};"
          f" при delta=1.1*delta*: min det = {g(1.1 * delta):.6f}")

    # ---------- Пункт 2: построение A2' --------------------------------
    A = IntervalMatrix(MID_A2, delta * RAD2_1)
    show_interval_matrix(A, "A2(delta*)")
    Ap = MID_A2 - delta * np.outer(best_y, best_z)
    print("\nПостроенная матрица A2' = mid A2 - delta* * y z^T:\n", Ap)

    # ---------- Пункт 3: проверка --------------------------------------
    print("\nПроверка:")
    print("  A2' in A2 :", A.contains(Ap))
    print("  |A2' - mid A2| <= delta* * E :", np.all(np.abs(Ap - MID_A2) <= delta + TOL),
          "(все элементы лежат на границе: |a'_ij - a_ij| = delta*)")
    print(f"  det A2' = {np.linalg.det(Ap):.3e}")
    sv = sing_values(Ap)
    print(f"  сингулярные числа: {sv}  -> rank = {np.linalg.matrix_rank(Ap, tol=1e-10)}")
    x = Ainv @ best_y
    print(f"  ненулевой x = (mid A2)^(-1) y = {x}; A2' x = {Ap @ x}")
    return dict(delta=delta, Ap=Ap, A=A, y=best_y, z=best_z, ds=ds, dmin=dmin, dmax=dmax, x=x, mu=mu)


# =====================================================================
# 3. Рисунки
# =====================================================================
def make_figures(r_regr, r_tomo, r_sq):
    # ---- Рис. 1: Зависимость границ параметра t от delta (томография) и интервалы (регрессия)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    d = np.linspace(0.0, 0.09, 400)
    cmax, cmin = 1.10, 0.95
    ax[0].plot(d, (cmax - d) / (1 + d), label=r"$t_{\min}(\delta)=\dfrac{1.10-\delta}{1+\delta}$")
    ax[0].plot(d, (cmin + d) / (1 - d), label=r"$t_{\max}(\delta)=\dfrac{0.95+\delta}{1-\delta}$")
    ax[0].axvline(r_tomo["delta"], color="k", ls="--", lw=1)
    ax[0].plot([r_tomo["delta"]], [r_tomo["t"]], "ro")
    ax[0].annotate(rf"$\delta^*={r_tomo['delta']:.5f}$, $t={r_tomo['t']:.3f}$",
                   (r_tomo["delta"], r_tomo["t"]), xytext=(0.043, 1.10),
                   arrowprops=dict(arrowstyle="->"))
    ax[0].set_xlabel(r"$\delta$"); ax[0].set_ylabel("t (коэффициент пропорциональности столбцов)")
    ax[0].set_title("Томография: допустимые t = [t_min, t_max]\nпоявляются при δ ≥ δ*")
    ax[0].legend(); ax[0].grid(alpha=.3)

    dr = np.linspace(0.0, 0.12, 400)
    c = MID_A1[:, 0]
    ax[1].plot(dr, c.max() - dr, label=r"$\max_i c_i-\delta$ (нижняя граница общей точки)")
    ax[1].plot(dr, c.min() + dr, label=r"$\min_i c_i+\delta$ (верхняя граница общей точки)")
    ax[1].axvline(r_regr["delta"], color="k", ls="--", lw=1)
    ax[1].plot([r_regr["delta"]], [r_regr["t"]], "ro")
    ax[1].annotate(rf"$\delta^*={r_regr['delta']:.3f}$, $t={r_regr['t']:.3f}$",
                   (r_regr["delta"], r_regr["t"]), xytext=(0.085, 1.07),
                   arrowprops=dict(arrowstyle="->"))
    ax[1].set_xlabel(r"$\delta$"); ax[1].set_ylabel("общая точка интервалов 1-го столбца")
    ax[1].set_title("Регрессия: пересечение интервалов\nпоявляется при δ ≥ δ*")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig1.png", dpi=170); plt.close(fig)

    # ---- Рис. 2: строки A1 как прямоугольники (a_i1, a_i2) и особая матрица
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, (r, ttl, regr) in enumerate([(r_regr, "Регрессия, δ = δ*", True), (r_tomo, "Томография, δ = δ*", False)]):
        A, Ap, dl = r["A"], r["Ap"], r["delta"]
        for i in range(3):
            x0, x1 = A.inf[i, 0], A.sup[i, 0]
            y0, y1 = A.inf[i, 1], A.sup[i, 1]
            if regr:
                ax[k].plot([x0, x1], [1 + 0.004 * (i - 1)] * 2, lw=6, alpha=.45, color=f"C{i}",
                           solid_capstyle="butt", label=f"строка {i + 1}")
            else:
                ax[k].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=.3, color=f"C{i}", label=f"строка {i + 1}"))
            ax[k].plot(Ap[i, 0], Ap[i, 1], "ko", ms=5)
            ax[k].plot(A.mid[i, 0], A.mid[i, 1], "x", color="gray")
        # прямая через начало координат: a2 = a1 / t
        xs = np.linspace(0.9, 1.15, 10)
        ax[k].plot(xs, xs / r["t"], "r--", lw=1, label="прямая через 0 (ранг 1)")
        ax[k].set_xlim(0.9, 1.15); ax[k].set_ylim(0.9, 1.1)
        ax[k].set_xlabel("$a_{i1}$"); ax[k].set_ylabel("$a_{i2}$")
        ax[k].set_title(ttl + "\n● — строки A1′ (на одной прямой через 0), × — центры", fontsize=10)
        ax[k].legend(fontsize=8); ax[k].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig2.png", dpi=170); plt.close(fig)

    # ---- Рис. 3: sigma_min / проверка — зависимость min delta(x) от направления x (прямоугольная A1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    th = np.linspace(0, np.pi, 4000)
    for rad1, nm in [(RAD1_REGR, "регрессия"), (RAD1_TOMO, "томография")]:
        vals = [oettli_prager_delta(np.array([np.cos(t), np.sin(t)]), MID_A1, rad1) for t in th]
        ax.plot(th, vals, label=nm)
    ax.set_ylim(0, 0.3); ax.set_xlim(2.0, 2.7)
    ax.axhline(r_regr["delta"], color="C0", ls=":"); ax.axhline(r_tomo["delta"], color="C1", ls=":")
    ax.set_xlabel(r"угол $\theta$ вектора $x=(\cos\theta,\sin\theta)$")
    ax.set_ylabel(r"$\delta(x)=\max_i |(\mathrm{mid}A\,x)_i|/(\mathrm{rad}_1|x|)_i$")
    ax.set_title("Минимум функции δ(x) даёт δ* (критерий Оеттли–Прагера)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig3.png", dpi=170); plt.close(fig)

    # ---- Рис. 4: диапазон det по вершинам A2(delta)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.fill_between(r_sq["ds"], r_sq["dmin"], r_sq["dmax"], alpha=.3, label="[min det, max det] по вершинам")
    ax.plot(r_sq["ds"], r_sq["dmin"], "C0"); ax.plot(r_sq["ds"], r_sq["dmax"], "C0")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(r_sq["delta"], color="r", ls="--", label=rf"$\delta^*={r_sq['delta']:.5f}$")
    ax.set_xlabel(r"$\delta$"); ax.set_ylabel(r"$\det A$")
    ax.set_title("A2: диапазон определителя; при δ ≥ δ* он содержит 0")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig4.png", dpi=170); plt.close(fig)


# =====================================================================
if __name__ == "__main__":
    print("ЛАБОРАТОРНАЯ РАБОТА №1: особенность интервальных матриц\n")
    r_tomo = analyze_rectangular("томография", RAD1_TOMO)
    print()
    r_regr = analyze_rectangular("регрессия", RAD1_REGR)
    r_sq = analyze_square()
    make_figures(r_regr, r_tomo, r_sq)

    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    print(f"A1, томография : delta* = {r_tomo['delta']:.8f} (= 1/27)")
    print(f"A1, регрессия  : delta* = {r_regr['delta']:.8f} (= 0.075)")
    print(f"A2, 3x3        : delta* = {r_sq['delta']:.8f}")
    print(f"Отношение delta*(регрессия)/delta*(томография) = {r_regr['delta'] / r_tomo['delta']:.4f}")
