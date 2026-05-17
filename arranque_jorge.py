import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from difraciones import *
# CONFIGURACIÓN GENERAL
Path('figuras').mkdir(exist_ok=True)
Path('data').mkdir(exist_ok=True)
M = 512
lmda = 600e-9
ua = 40
z = ua * 1.496e11
plano = calc_plano(3000, lmda, ua)

# DEFINICIÓN DEL OBJETO ELÍPTICO
def elipse(a, b, theta=0):
    x = np.linspace(-plano/2, plano/2, M)
    X, Y = np.meshgrid(x, x)
    ct, st = np.cos(theta), np.sin(theta)
    Xr = ct*X + st*Y
    Yr = -st*X + ct*Y
    U = np.ones((M, M))
    U[(Xr/a)**2 + (Yr/b)**2 <= 1] = 0
    return U

# PERFIL DE DIFRACCIÓN
def perfil(a, b, theta, impacto):
    I = fresnel(elipse(a, b, theta), M, plano, z, lmda)
    x, y = extraer_perfil(I, M, plano, 0, impacto)
    return x, y / np.median(y[:50])

# CURVA ORIGINAL DEL USUARIO
x, y = perfil(1800, 1100, 0.5, 300)
rng = np.random.default_rng(123)
yobs = y + rng.normal(0, 0.03, y.size)
np.savetxt(
    'data/curva_sintetica.csv',
    np.c_[x, yobs],
    delimiter=',',
    header='x_m,flujo',
    comments=''
)
best = None
for a in [1400, 1800, 2200]:
    for b in [900, 1100, 1400]:
        for th in [0, 0.5, 1.0]:
            for imp in [0, 300, 600]:
                xm, ym = perfil(a, b, th, imp)
                chi2 = np.mean(((yobs - ym) / 0.03)**2)
                if best is None or chi2 < best[0]:
                    best = (chi2, a, b, th, imp, ym)


# GRÁFICA ORIGINAL
plt.figure(figsize=(9,6))
plt.plot(x/1000, yobs, '.', label='datos')
plt.plot(
    x/1000,
    best[5],
    linewidth=2,
    label=f'mejor: chi2={best[0]:.1f}, a={best[1]}, b={best[2]}, th={best[3]}, impacto={best[4]}'
)
plt.xlabel('posición [km]')
plt.ylabel('flujo')
plt.title('Ajuste original')
plt.grid()
plt.legend()
plt.savefig('figuras/ajuste_original.png', dpi=150)
plt.show()


# 1) DISTINTOS TAMAÑOS
parametros = [
    (500, 500),
    (1000, 700),
    (1800, 1100),
    (3000, 2000)
]
for a, b in parametros:
    xk, yk = perfil(a, b, 0.5, 300)
    plt.figure(figsize=(9,6))
    plt.plot(xk/1000, yk, linewidth=2)
    plt.xlabel('posición [km]')
    plt.ylabel('flujo')
    plt.title(f'Curva de luz — tamaño a={a} m, b={b} m')
    plt.grid()
    plt.savefig(f'figuras/tamano_a{a}_b{b}.png', dpi=150)
    plt.show()

# 2) DISTINTOS IMPACTOS
for imp in [0, 300, 600, 900]:
    xk, yk = perfil(1800, 1100, 0.5, imp)
    plt.figure(figsize=(9,6))
    plt.plot(xk/1000, yk, linewidth=2)
    plt.xlabel('posición [km]')
    plt.ylabel('flujo')
    plt.title(f'Curva de luz — impacto = {imp} m')
    plt.grid()
    plt.savefig(f'figuras/impacto_{imp}.png', dpi=150)
    plt.show()

# 3) DISTINTAS ORIENTACIONES
for th in [0, 0.5, 1.0, 1.5]:
    xk, yk = perfil(1800, 1100, th, 300)
    plt.figure(figsize=(9,6))
    plt.plot(xk/1000, yk, linewidth=2)
    plt.xlabel('posición [km]')
    plt.ylabel('flujo')
    plt.title(f'Curva de luz — theta = {th:.1f} rad')
    plt.grid()
    plt.savefig(f'figuras/theta_{th:.1f}.png', dpi=150)
    plt.show()

# 4) MAPA 2D DE DIFRACCIÓN
I = fresnel(elipse(1800, 1100, 0.5), M, plano, z, lmda)
x2 = np.linspace(-plano/2, plano/2, M)/1000
plt.figure(figsize=(8,8))
plt.imshow(
    I,
    extent=[x2.min(), x2.max(), x2.min(), x2.max()],
    origin='lower'
)
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.title('Patrón 2D de difracción Fresnel')
plt.colorbar(label='intensidad')
plt.savefig('figuras/patron_2D.png', dpi=150)
plt.show()
print('\nSimulación terminada.')
print('Todas las figuras fueron guardadas en la carpeta "figuras".')