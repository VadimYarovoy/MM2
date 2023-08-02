import math
from typing import Callable, Tuple

from grid import *
from sparse_matrix import *
from linalg import *

real1_fn = Callable[[float], float]
real2_fn = Callable[[float, float], float]


def _cholesky_decomp(A, Nx, Ny):
    n = (Nx + 1) * (Ny + 1)
    m = Nx + 1

    a = [A.get(i, i) for i in range(n)]
    b = [A.get(i, i + 1) for i in range(n - 1)]
    c = [A.get(i, i + m) for i in range(n - m)]

    a_new = []
    b_new = []
    c_new = []

    L = SparseMatrixCOO()

    for i in range(n):

        if i - 1 < 0:
            ai = math.sqrt(a[i])
        elif i - m < 0:
            ai = math.sqrt(a[i] - b_new[i - 1]**2)
        else:
            ai = math.sqrt(a[i] - b_new[i - 1]**2 - c_new[i - m]**2)

        a_new.append(ai)

        if i < n - 1:
            bi = b[i] / ai
            b_new.append(bi)

        if i < n - m:
            ci = c[i] / ai
            c_new.append(ci)

    for i in range(n):
        L.set(i, i, a_new[i])

    for i in range(n - 1):
        L.set(i, i + 1, b_new[i])

    for i in range(n - m):
        L.set(i, i + m, c_new[i])

    return transpose(L)


class Model:
    def __init__(self,
                 a: float, b: float,
                 c: float, d: float,
                 k: Tuple[real2_fn],
                 f: real2_fn,
                 chi: Tuple[float],
                 g: Tuple[real1_fn]) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k
        self.f = f
        self.chi = chi
        self.g = g

    def init_data(self, Nx: int, Ny: int):

        k1, k2 = self.k
        f = self.f

        chi1, chi3, chi4 = self.chi
        g1, g2, g3, g4 = self.g

        hx = (self.b - self.a) / Nx
        hy = (self.d - self.c) / Ny

        x1, x2 = get_x(self.a, self.b, Nx)
        y1, y2 = get_y(self.c, self.d, Ny)

        M = (Nx + 1) * (Ny + 1)
        L = (Nx + 1)

        mtrx = SparseMatrixCOO()
        g = [0] * M

        # inner space
        for i in range(1, Nx):
            for j in range(1, Ny):
                m = j * L + i

                a1 = (hx / hy) * k2( x1(i), y2(j - 1) )
                a2 = (hy / hx) * k1( x2(i - 1), y1(j) )
                a3 = (hy / hx) * k1( x2(i), y1(j) )
                a4 = (hx / hy) * k2( x1(i), y2(j) )

                mtrx.set(m, m - L, -a1)
                mtrx.set(m, m - 1, -a2)
                mtrx.set(m, m, a1 + a2 + a3 + a4)
                mtrx.set(m, m + 1, -a3)
                mtrx.set(m, m + L, -a4)
                g[m] = hx * hy * f(x1(i), y1(j))

        # left edge
        i = 0
        for j in range(1, Ny):
            m = j * L + i

            a1 = (hx / hy / 2) * k2( x1(i), y2(j - 1) )
            a2 = hy * chi1
            a3 = (hy / hx) * k1( x2(i), y1(j) )
            a4 = (hx / hy / 2) * k2( x1(i), y2(j) )

            mtrx.set(m, m - L, -a1)
            mtrx.set(m, m, a1 + a2 + a3 + a4)
            mtrx.set(m, m + 1, -a3)
            mtrx.set(m, m + L, -a4)
            g[m] = (hx / 2) * hy * f(x1(i), y1(j)) + hy * g1(y1(j))

        # right edge
        i = Nx
        for j in range(1, Ny):
            m = j * L + i

            mtrx.set(m, m, 1)
            g[m] = g2(y1(j))

        # bottom edge
        j = 0
        for i in range(1, Nx):
            m = j * L + i

            a1 = hx * chi3
            a2 = (hy / hx / 2) * k1( x2(i - 1), y1(j) )
            a3 = (hy / hx / 2) * k1( x2(i), y1(j) )
            a4 = (hx / hy) * k2( x1(i), y2(j) )

            mtrx.set(m, m - 1, -a2)
            mtrx.set(m, m, a1 + a2 + a3 + a4)
            mtrx.set(m, m + 1, -a3)
            mtrx.set(m, m + L, -a4)
            g[m] = hx * (hy / 2) * f(x1(i), y1(j)) + hx * g3(x1(i))

        # top edge
        j = Ny
        for i in range(1, Nx):
            m = j * L + i

            a1 = (hx / hy) * k2( x1(i), y2(j - 1) )
            a2 = (hy / hx / 2) * k1( x2(i - 1), y1(j) )
            a3 = (hy / hx / 2) * k1( x2(i), y1(j) )
            a4 = hx * chi4

            mtrx.set(m, m - L, -a1)
            mtrx.set(m, m - 1, -a2)
            mtrx.set(m, m, a1 + a2 + a3 + a4)
            mtrx.set(m, m + 1, -a3)
            g[m] = hx * (hy / 2) * f(x1(i), y1(j)) + hx * g4(x1(i))

        # left bottom
        i, j = 0, 0
        m = j * L + i

        a1 = (hx / 2) * chi3
        a2 = (hy / 2) * chi1
        a3 = (hy / hx / 2) * k1(x2(i), y1(j))
        a4 = (hx / hy / 2) * k2(x1(i), y2(j))

        mtrx.set(m, m, a1 + a2 + a3 + a4)
        mtrx.set(m, m + 1, -a3)
        mtrx.set(m, m + L, -a4)
        g[m] = (hx / 2) * (hy / 2) * f(x1(i), y1(j)) + (hy / 2) * g1(y1(j)) + (hx / 2) * g3(x1(i))

        # right bottom
        i, j = Nx, 0
        m = j * L + i

        mtrx.set(m, m, 1)
        g[m] = g2(y1(j))

        # right top corner
        i, j = Nx, Ny
        m = j * L + i

        mtrx.set(m, m, 1)
        g[m] = g2(y1(j))

        # left top corner
        i, j = 0, Ny
        m = j * L + i

        a1 = (hx / hy / 2) * k2(x1(i), y2(j - 1))
        a2 = (hy / 2) * chi1
        a3 = (hy / hx / 2) * k1(x2(i), y1(j))
        a4 = (hx / 2) * chi4

        mtrx.set(m, m - L, -a1)
        mtrx.set(m, m, a1 + a2 + a3 + a4)
        mtrx.set(m, m + 1, -a3)
        g[m] = (hx / 2) * (hy / 2) * f(x1(i), y1(j)) + (hy / 2) * g1(y1(j)) + (hx / 2) * g4(x1(i))

        # remove extra elements
        i = Nx - 1
        for j in range(1, Ny):
            m = j * L + i

            x = mtrx.get(m, m + 1)
            mtrx.set(m, m + 1, 0)
            g[m] -= g[m + 1] * x

        i, j = Nx - 1, 0
        m = j * L + i

        x = mtrx.get(m, m + 1)
        mtrx.set(m, m + 1, 0)
        g[m] -= g[m + 1] * x

        i, j = Nx - 1, Ny
        m = j * L + i

        x = mtrx.get(m, m + 1)
        mtrx.set(m, m + 1, 0)
        g[m] -= g[m + 1] * x

        L = _cholesky_decomp(mtrx, Nx, Ny)

        mtrx = mtrx.to_symmetric().to_CSR()
        L = L.to_CSR()

        return (mtrx, g, L)