import json
import math
import numpy as np
from scipy.optimize import brentq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = math.pi

#Интервальная арифметика
class I:
    def __init__(self, lo, hi=None):
        hi = lo if hi is None else hi
        assert lo <= hi + 1e-15
        self.lo, self.hi = float(lo), float(hi)

    def __add__(self, o):
        o = _i(o)
        return I(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, o):
        o = _i(o)
        return I(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        return _i(o) - self

    def __mul__(self, o):
        o = _i(o)
        p = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return I(min(p), max(p))

    __rmul__ = __mul__

    def sqr(self):
        if self.lo >= 0:
            return I(self.lo ** 2, self.hi ** 2)
        if self.hi <= 0:
            return I(self.hi ** 2, self.lo ** 2)
        return I(0.0, max(self.lo ** 2, self.hi ** 2))

    def cube(self):
        return I(self.lo ** 3, self.hi ** 3)

    @property
    def mid(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def rad(self):
        return 0.5 * (self.hi - self.lo)

    @property
    def mag(self):
        return max(abs(self.lo), abs(self.hi))

    def contains(self, v, tol=1e-12):
        return self.lo - tol <= v <= self.hi + tol

    def as_list(self):
        return [self.lo, self.hi]

    def __repr__(self):
        return f"[{self.lo:.6f}; {self.hi:.6f}]"


def _i(x):
    return x if isinstance(x, I) else I(x, x)


def hull(*xs):
    """Интервальная оболочка объединения: [min lo, max hi]."""
    return I(min(x.lo for x in xs), max(x.hi for x in xs))


def hausdorff(A, B):
    """Расстояние по Хаусдорфу между интервалами: max(|a_lo - b_lo|, |a_hi - b_hi|)."""
    return max(abs(A.lo - B.lo), abs(A.hi - B.hi))


def isin(X):
    a, b = X.lo, X.hi
    if b - a >= 2 * PI:
        return I(-1, 1)
    lo, hi = min(math.sin(a), math.sin(b)), max(math.sin(a), math.sin(b))
    if PI / 2 + 2 * PI * math.ceil((a - PI / 2) / (2 * PI)) <= b:
        hi = 1.0
    if -PI / 2 + 2 * PI * math.ceil((a + PI / 2) / (2 * PI)) <= b:
        lo = -1.0
    return I(lo, hi)


def icos(X):
    a, b = X.lo, X.hi
    if b - a >= 2 * PI:
        return I(-1, 1)
    lo, hi = min(math.cos(a), math.cos(b)), max(math.cos(a), math.cos(b))
    if 2 * PI * math.ceil(a / (2 * PI)) <= b:
        hi = 1.0
    if PI + 2 * PI * math.ceil((a - PI) / (2 * PI)) <= b:
        lo = -1.0
    return I(lo, hi)


# критические точки sinc(u) = sin(u)/u: корни u cos u = sin u (u = ±4.4934, ±7.7253, ...)
_SINC_CRIT = []
for _k in range(1, 8):
    _r = brentq(lambda u: u * math.cos(u) - math.sin(u), _k * PI + 1e-9, _k * PI + PI / 2 - 1e-9)
    _SINC_CRIT += [_r, -_r]

def _sinc(u):
    return 1.0 if abs(u) < 1e-14 else math.sin(u) / u


def isinc(X):
    pts = [X.lo, X.hi] + [u for u in _SINC_CRIT if X.lo <= u <= X.hi]
    if X.lo <= 0 <= X.hi:
        pts.append(0.0)
    v = [_sinc(u) for u in pts]
    return I(min(v), max(v))


# Общие методы оценивания (одна переменная)
def fmv(f, dF, c, X):
    """Дифференциальная центрированная форма Fmv(X, c) = f(c) + F'(X)(X - c)."""
    return f(c) + dF * (X - c)


def fsl(f, S, c, X):
    """Наклонная центрированная форма Fsl(X, c) = f(c) + S(X, c)(X - c); S - интервальный наклон."""
    return f(c) + S(c, X) * (X - c)


def baumann_centers(D, X):
    """
    Центры Бауманна для D = F'(X) = [d_lo, d_hi], X = [x_lo, x_hi]:
        c_lo = (d_hi x_lo - d_lo x_hi)/(d_hi - d_lo)   - оптимален для нижней границы,
        c_hi = (d_hi x_hi - d_lo x_lo)/(d_hi - d_lo)   - оптимален для верхней границы.
    Если 0 не принадлежит D (функция монотонна), центры - концы X.
    """
    if D.lo >= 0:
        return X.lo, X.hi
    if D.hi <= 0:
        return X.hi, X.lo
    w = D.hi - D.lo
    return (D.hi * X.lo - D.lo * X.hi) / w, (D.hi * X.hi - D.lo * X.lo) / w


def bicentered(f, D, X):
    """Бицентрированная среднезначная форма: [нижняя граница Fmv(X, c_lo), верхняя граница Fmv(X, c_hi)]."""
    c_lo, c_hi = baumann_centers(D, X)
    return I(fmv(f, D, c_lo, X).lo, fmv(f, D, c_hi, X).hi), c_lo, c_hi


def exact_range_1d(f, df, a, b, n=200001):
    """Точная область значений: экстремумы в концах и в корнях f' (корни ищутся brentq по смене знака)."""
    xs = np.linspace(a, b, n)
    g = df(xs)
    pts = [a, b]
    for i in np.where(g[:-1] * g[1:] < 0)[0]:
        pts.append(brentq(df, xs[i], xs[i + 1], xtol=1e-14))
    for i in np.where(g == 0)[0]:
        pts.append(xs[i])
    vals = [f(p) for p in pts]
    return I(min(vals), max(vals)), pts[int(np.argmin(vals))], pts[int(np.argmax(vals))], sorted(set(pts))


def lipschitz_1d(df, ddf, a, b, n=200001):
    """L = max |f'| на [a, b]: максимум |f'| достигается в концах или в корнях f''."""
    xs = np.linspace(a, b, n)
    g = ddf(xs)
    pts = [a, b]
    for i in np.where(g[:-1] * g[1:] < 0)[0]:
        pts.append(brentq(ddf, xs[i], xs[i + 1], xtol=1e-14))
    vals = [abs(df(p)) for p in pts]
    k = int(np.argmax(vals))
    return vals[k], pts[k]


def analyze_1d(name, f, df, ddf, X, natural, alternatives, dF, slope, centers_extra):
    """Пункты A-C для функции одной переменной. Возвращает словарь результатов."""
    a, b = X.lo, X.hi
    ran, x_min, x_max, crit = exact_range_1d(f, df, a, b)
    L, x_L = lipschitz_1d(df, ddf, a, b)
    bound = L * X.rad
    D = dF(X)

    rows = []

    def add(code, label, F, extra=None):
        r = dict(code=code, label=label, enc=F.as_list(), rad=F.rad, haus=hausdorff(F, ran),
                 ok=bool(F.rad <= bound + 1e-12))
        if extra:
            r.update(extra)
        rows.append(r)

    add("B.1", "естественное расширение", natural(X))
    for lab, fn in alternatives:
        add("B.2", lab, fn(X))
    centers = [("c = a", a), ("c = mid X", X.mid), ("c = b", b), (f"c = x_min ≈ {x_min:.4f}", x_min)] + centers_extra
    for lab, c in centers:
        add("B.3", f"Fmv, {lab}", fmv(f, D, c, X), dict(c=c))
    for lab, c in centers:
        add("B.4", f"Fsl, {lab}", fsl(f, slope, c, X), dict(c=c))
    Fbc, c_lo, c_hi = bicentered(f, D, X)
    add("B.5", "бицентрированная ", Fbc, dict(c_lo=c_lo, c_hi=c_hi))

    res = dict(name=name, X=X.as_list(), ran=ran.as_list(), x_min=x_min, x_max=x_max, crit=crit,
               L=L, x_L=x_L, radX=X.rad, bound=bound, dF=D.as_list(), rows=rows)
    print("\n")
    print(f"{name},  X = {X}")
    print(f"A. точная область значений ran(f, X) = {ran}   (min в x = {x_min:.6f}, max в x = {x_max:.6f})")
    print(f"   критические точки и концы: {[round(p, 6) for p in crit]}")
    print(f"   F'(X) = {D}")
    print(f"C.1 L = max|f'| = {L:.6f} (в x = {x_L:.4f});  C.2 L*rad X = {bound:.6f};  rad ran = {ran.rad:.6f}")
    print(f"{'пункт':5} {'метод':34} {'F(X)':30} {'rad F':>9} {'dist':>9} {'rad<=L rad X':>12}")
    for r in rows:
        print(f"{r['code']:5} {r['label']:34} [{r['enc'][0]:10.5f}; {r['enc'][1]:10.5f}] "
              f"{r['rad']:9.4f} {r['haus']:9.4f} {str(r['ok']):>12}")
    return res

#f1(x) = x^3 - 3x^2 + 2,  X = [0, 3]
def f1(x):
    return x ** 3 - 3 * x ** 2 + 2


def df1(x):
    return 3 * x ** 2 - 6 * x


def ddf1(x):
    return 6 * x - 6


def F1_natural(X):
    return X.cube() - 3 * X.sqr() + 2


def F1_horner(X):
    return ((X - 3) * X) * X + 2                 # x^3 - 3x^2 + 2 = ((x - 3) x) x + 2


def F1_shift(X):
    T = X - 1                                    # x^3 - 3x^2 + 2 = t^3 - 3t,  t = x - 1
    return T.cube() - 3 * T


def dF1(X):
    return (3 * X - 6) * X                       # производная в форме Горнера: (3x - 6) x


def slope1(c, X):
    """f1(x) - f1(c) = (x - c) q(x),  q(x) = x^2 + (c - 3) x + c (c - 3); q - в форме Горнера."""
    return (X + (c - 3)) * X + c * (c - 3)

# f2(x) = x^2 + cos(3x),  X = [-1, 2]
def f2(x):
    return x ** 2 + np.cos(3 * x)


def df2(x):
    return 2 * x - 3 * np.sin(3 * x)


def ddf2(x):
    return 2 - 9 * np.cos(3 * x)


def F2_natural(X):
    return X.sqr() + icos(3 * X)


def F2_cheb(X):
    c = icos(X)                                  # cos 3x = 4cos^3 x - 3cos x = cos x (4cos^2 x - 3)
    return X.sqr() + c * (4 * c.sqr() - 3)


def dF2(X):
    return 2 * X - 3 * isin(3 * X)


def slope2(c, X):
    """
    Наклон f2: (x^2 - c^2)/(x - c) = x + c;
    (cos 3x - cos 3c)/(x - c) = -3 sin(1.5 (x + c)) * sinc(1.5 (x - c)),  sinc(u) = sin(u)/u.
    """
    return (X + c) - 3 * isin(1.5 * (X + c)) * isinc(1.5 * (X - c))


# f3(x, y) = 2x^2 + 3y^2 - 2xy + x - 2y,  X = [-2, 1] x [-1, 2]
def f3(x, y):
    return 2 * x ** 2 + 3 * y ** 2 - 2 * x * y + x - 2 * y


def dfx3(x, y):
    return 4 * x - 2 * y + 1


def dfy3(x, y):
    return 6 * y - 2 * x - 2


def F3_natural(X, Y):
    return 2 * X.sqr() + 3 * Y.sqr() - 2 * (X * Y) + X - 2 * Y


def F3_sq1(X, Y):                                # 2x^2 + 3y^2 - 2xy = (x - y)^2 + x^2 + 2y^2
    return (X - Y).sqr() + X.sqr() + 2 * Y.sqr() + X - 2 * Y


def F3_sq2(X, Y):                                # полные квадраты: 2(x - y/2 + 1/4)^2 + 2.5(y - 0.3)^2 - 0.35
    return 2 * (X - 0.5 * Y + 0.25).sqr() + 2.5 * (Y - 0.3).sqr() - 0.35


def dFx3(X, Y):
    return 4 * X - 2 * Y + 1


def dFy3(X, Y):
    return 6 * Y - 2 * X - 2


def fmv2(c, X, Y):
    """Fmv(X, c) = f3(c) + df3/dx(X) (X1 - c1) + df3/dy(X) (X2 - c2)."""
    return f3(*c) + dFx3(X, Y) * (X - c[0]) + dFy3(X, Y) * (Y - c[1])


def bicentered2(X, Y):
    """Бицентрированная форма (теорема Бауманна): покомпонентные центры c_lo, c_hi."""
    Dx, Dy = dFx3(X, Y), dFy3(X, Y)
    cx_lo, cx_hi = baumann_centers(Dx, X)
    cy_lo, cy_hi = baumann_centers(Dy, Y)
    c_lo, c_hi = (cx_lo, cy_lo), (cx_hi, cy_hi)
    return I(fmv2(c_lo, X, Y).lo, fmv2(c_hi, X, Y).hi), c_lo, c_hi


def analyze_f3():
    X, Y = I(-2, 1), I(-1, 2)
    print("\n")
    print(f"f3(x, y) = 2x^2 + 3y^2 - 2xy + x - 2y,  X = {X} x {Y}")

    H = np.array([[4.0, -2.0], [-2.0, 6.0]])
    star = np.linalg.solve(H, -np.array([1.0, -2.0]))             # grad f3 = H z + (1, -2) = 0
    inside = X.contains(star[0]) and Y.contains(star[1])
    verts = [(x, y) for x in (X.lo, X.hi) for y in (Y.lo, Y.hi)]
    vv = [f3(*v) for v in verts]
    v_max = verts[int(np.argmax(vv))]
    ran = I(f3(*star), max(vv))
    xs, ys = np.meshgrid(np.linspace(X.lo, X.hi, 1201), np.linspace(Y.lo, Y.hi, 1201))
    gv = f3(xs, ys)
    print(f"E.2 собств. числа Гессе = {np.linalg.eigvalsh(H)} (>0, f3 выпукла); стационарная точка = {star}, внутри бруса: {inside}")
    print(f"    значения в вершинах: {dict(zip(map(str, verts), vv))}")
    print(f"    ran(f3, X) = {ran};  max в {v_max};  проверка по сетке 1201x1201: [{gv.min():.6f}; {gv.max():.6f}]")

    L1 = max(abs(dfx3(*v)) for v in verts)
    L2 = max(abs(dfy3(*v)) for v in verts)
    bound = L1 * X.rad + L2 * Y.rad
    print(f"E.8 L1 = sup|df/dx| = {L1}, L2 = sup|df/dy| = {L2};  L1 rad X1 + L2 rad X2 = {bound}")

    rows = []

    def add(code, label, F, extra=None):
        r = dict(code=code, label=label, enc=F.as_list(), rad=F.rad, haus=hausdorff(F, ran),
                 ok=bool(F.rad <= bound + 1e-12))
        if extra:
            r.update(extra)
        rows.append(r)

    add("E.3", "естественное расширение исходного выражения", F3_natural(X, Y))
    add("E.4", "(x − y)² + x² + 2y² + x − 2y", F3_sq1(X, Y))
    add("E.4", "2(x − y/2 + 1/4)² + 2,5(y − 0,3)² − 0,35", F3_sq2(X, Y))
    centers = [("c = mid X = (−0,5; 0,5)", (X.mid, Y.mid)), ("c = (−2; −1)", (X.lo, Y.lo)),
               ("c = (1; 2)", (X.hi, Y.hi)), ("c = argmin = (−0,1; 0,3)", (star[0], star[1]))]
    for lab, c in centers:
        add("E.5", f"Fmv, {lab}", fmv2(c, X, Y), dict(c=list(c)))
    Fbc, c_lo, c_hi = bicentered2(X, Y)
    add("E.6", "бицентрированная (Бауманн)", Fbc, dict(c_lo=list(c_lo), c_hi=list(c_hi)))

    def split_x():
        Xa, Xb = I(X.lo, X.mid), I(X.mid, X.hi)
        A, B = F3_natural(Xa, Y), F3_natural(Xb, Y)
        return Xa, Xb, A, B, hull(A, B)

    def split_y():
        Ya, Yb = I(Y.lo, Y.mid), I(Y.mid, Y.hi)
        A, B = F3_natural(X, Ya), F3_natural(X, Yb)
        return Ya, Yb, A, B, hull(A, B)

    sx, sy = split_x(), split_y()
    split = dict(
        x=dict(parts=[sx[0].as_list(), sx[1].as_list()], enc=[sx[2].as_list(), sx[3].as_list()], hull=sx[4].as_list(),
               rad=sx[4].rad, haus=hausdorff(sx[4], ran)),
        y=dict(parts=[sy[0].as_list(), sy[1].as_list()], enc=[sy[2].as_list(), sy[3].as_list()], hull=sy[4].as_list(),
               rad=sy[4].rad, haus=hausdorff(sy[4], ran)))
    rows.append(dict(code="E.7", label="дробление по x, оболочка", enc=sx[4].as_list(), rad=sx[4].rad,
                     haus=hausdorff(sx[4], ran), ok=bool(sx[4].rad <= bound + 1e-12)))
    rows.append(dict(code="E.7", label="дробление по y, оболочка", enc=sy[4].as_list(), rad=sy[4].rad,
                     haus=hausdorff(sy[4], ran), ok=bool(sy[4].rad <= bound + 1e-12)))

    print(f"F3'x(X) = {dFx3(X, Y)},  F3'y(X) = {dFy3(X, Y)}")
    print(f"центры Бауманна: c_lo = {c_lo}, c_hi = {c_hi}")
    print(f"{'пункт':5} {'метод':48} {'F(X)':30} {'rad F':>9} {'dist':>9} {'rad<=Σ L rad':>12}")
    for r in rows:
        print(f"{r['code']:5} {r['label']:48} [{r['enc'][0]:10.5f}; {r['enc'][1]:10.5f}] "
              f"{r['rad']:9.4f} {r['haus']:9.4f} {str(r['ok']):>12}")
    print("E.7 подбрусы по x:", sx[0], sx[1], "->", sx[2], sx[3], "-> оболочка", sx[4])
    print("E.7 подбрусы по y:", sy[0], sy[1], "->", sy[2], sy[3], "-> оболочка", sy[4])

    return dict(X=X.as_list(), Y=Y.as_list(), ran=ran.as_list(), star=list(star), v_max=list(v_max),
                vert_vals=[[list(v), val] for v, val in zip(verts, vv)], eig=list(np.linalg.eigvalsh(H)),
                dFx=dFx3(X, Y).as_list(), dFy=dFy3(X, Y).as_list(), L1=L1, L2=L2, radX=X.rad, radY=Y.rad,
                bound=bound, rows=rows, split=split, c_lo=list(c_lo), c_hi=list(c_hi),
                grid=[float(gv.min()), float(gv.max())])


def plot_1d(f, X, res, fname, title):
    xs = np.linspace(X.lo, X.hi, 1000)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, f(xs), lw=2, label="f(x)")
    ax.axhline(res["ran"][0], color="g", ls="--", lw=1, label=f"min f = {res['ran'][0]:.4f}")
    ax.axhline(res["ran"][1], color="r", ls="--", lw=1, label=f"max f = {res['ran'][1]:.4f}")
    ax.plot([res["x_min"]], [res["ran"][0]], "go")
    ax.plot([res["x_max"]], [res["ran"][1]], "ro")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title(title); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(fname, dpi=170); plt.close(fig)


def plot_f3(res):
    X, Y = I(*res["X"]), I(*res["Y"])
    xs, ys = np.meshgrid(np.linspace(X.lo, X.hi, 200), np.linspace(Y.lo, Y.hi, 200))
    zs = f3(xs, ys)
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(xs, ys, zs, cmap="viridis", alpha=.9, linewidth=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title("z = f3(x, y)")
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(xs, ys, zs, levels=18, cmap="viridis")
    ax2.clabel(cs, fontsize=7)
    ax2.plot([res["star"][0]], [res["star"][1]], "ro", label=f"min = {res['ran'][0]:.2f}")
    ax2.plot([res["v_max"][0]], [res["v_max"][1]], "ks", label=f"max = {res['ran'][1]:.0f}")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_title("Линии уровня на брусе X"); ax2.legend()
    ax2.set_aspect("equal"); ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_f3.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    X1, X2 = I(0, 3), I(-1, 2)

    r1 = analyze_1d("f1(x) = x^3 - 3x^2 + 2", f1, df1, ddf1, X1, F1_natural,
                    [("Горнер ((x − 3)x)x + 2", F1_horner), ("(x − 1)³ − 3(x − 1)", F1_shift)],
                    dF1, slope1, [])
    r2 = analyze_1d("f2(x) = x^2 + cos 3x", f2, df2, ddf2, X2, F2_natural,
                    [("x² + cos x (4cos² x − 3)", F2_cheb)],
                    dF2, slope2, [("c = 0 (локальный max)", 0.0)])
    r3 = analyze_f3()

    plot_1d(f1, X1, r1, "fig_f1.png", "f1(x) = x³ − 3x² + 2 на [0, 3]")
    plot_1d(f2, X2, r2, "fig_f2.png", "f2(x) = x² + cos 3x на [−1, 2]")
    plot_f3(r3)

    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(dict(f1=r1, f2=r2, f3=r3), fh, ensure_ascii=False, indent=1)
    print("\n рисунки сохранены")
