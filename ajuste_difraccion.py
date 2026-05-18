"""
HERRAMIENTA INTERACTIVA DE AJUSTE DE PERFILES DE DIFRACCIÓN
Basada en difraciones.py (Joel Castro 2019) y arranque.py
Integra: forma, tamaño, orientación, parámetro de impacto y chi-cuadrada
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ========== LIBRERÍA DE DIFRACCIÓN ==========

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return phi, rho

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def pupilCO(M, D, d):
    '''Generar obstrucción circular (apertura - obstrucción central)'''
    m = np.linspace(-D/2, D/2, M)
    a, b = np.meshgrid(m, m)
    th, r = cart2pol(a, b)
    P = np.double(r >= d/2)
    return P

def trasladar(P, smx, smy):
    '''Trasladar matriz en X, Y en píxeles'''
    MM = np.zeros(P.shape)
    x, y = MM.shape
    x = int(x/2)
    y = int(y/2)
    mx = int(smx)
    my = int(smy)
    
    if my == 0 or type(smy) == float:
        MM = P
    if my > 0:
        MM[y+my:, :] = P[y:-my, :]
        MM[:my, :] = P[2*y-my:, :]
        MM[my:y+my, :] = P[:y, :]
    if my < 0:
        MM[:y+my, :] = P[-my:y, :]
        MM[2*y+my:, :] = P[:-my, :]
        MM[y+my:2*y+my, :] = P[y:, :]
    
    M2 = MM.copy()
    
    if mx == 0 or type(smx) == float:
        M2 = MM
    
    if mx > 0:
        M2[:, x+mx:] = MM[:, x:-mx]
        M2[:, :mx] = MM[:, 2*x-mx:]
        M2[:, mx:x+mx] = MM[:, :x]
    if mx < 0:
        M2[:, :x+mx] = MM[:, -mx:x]
        M2[:, 2*x+mx:] = MM[:, :-mx]
        M2[:, x+mx:2*x+mx] = MM[:, x:]
    
    return M2

def pupil_doble(M, D, d):
    '''Generar obstrucción tipo binario de contacto'''
    r1 = (d/2) * 0.65
    r2 = np.sqrt((d/2)**2 - (r1)**2)
    d1 = r1 * 2
    d2 = r2 * 2
    Dx = 0.45 * d1 + 0.45 * d2
    Dy = 0
    sepX = ((Dx/2)/D) * M
    sepY = ((Dy/2)/D) * M
    
    m = np.linspace(-D/2, D/2, M)
    a, b = np.meshgrid(m, m)
    th, r = cart2pol(a, b)
    
    P1 = np.double(r >= r1)
    P2 = np.double(r >= r2)
    P = trasladar(P1, -sepX, sepY) + trasladar(P2, sepX, sepY)
    P = P == 2
    
    return P

def pupilSO(M, D, d):
    '''Generar obstrucción cuadrada'''
    t = (M * d / D)
    c = M / 2
    P = np.ones((M, M))
    P[int(-t/2+c):int(t/2+c), int(-t/2+c):int(t/2+c)] = 0
    return P

def fresnel(U0, M, plano, z, lmda):
    '''Calcular patrón de difracción de Fresnel (intensidad)'''
    k = 2 * np.pi / lmda
    nx, ny = np.shape(U0)
    x = (plano / M) * nx
    y = (plano / M) * ny
    fx = 1 / x
    fy = 1 / y
    
    u = np.ones((nx, 1)) * (np.arange(0, nx) - nx/2) * fx
    v = np.transpose((np.arange(0, ny) - ny/2) * np.ones((ny, 1)) * fy)
    
    O = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U0)))
    H = np.exp(1j*k*z) * np.exp(-1j*np.pi*(lmda*z)*(u**2 + v**2))
    
    U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.multiply(O, H))))
    I = np.abs(U)**2
    return I

def extraer_perfil(I0, M, D, T, b):
    '''Extraer perfil 1D del patrón 2D de difracción'''
    m2p = M / D
    x = np.linspace(-D/2, D/2, M)
    
    x1 = x * np.cos(T*np.pi/180) - b * np.sin(T*np.pi/180)
    x2 = x * np.sin(T*np.pi/180) + b * np.cos(T*np.pi/180)
    
    hp = np.array(m2p * x1) + M/2
    vp = np.array(m2p * x2) + M/2
    hp = hp.astype(int)
    vp = vp.astype(int)
    
    # Limitar índices
    hp = np.clip(hp, 0, M-1)
    vp = np.clip(vp, 0, M-1)
    
    y = np.ones(x.shape)
    for k in range(M-1):
        y[k] = I0[vp[k], hp[k]]
    
    return x, y

def calc_plano(d, lmda, ua):
    '''Calcular tamaño óptimo del plano de difracción'''
    z = ua * 1.496e11
    fscale = np.sqrt(lmda * z / 2)
    Rho = d / (2 * fscale)
    plano = (50 * d) / Rho
    return plano

def add_ruido_poisson(I, snr_factor=0.03):
    '''Añadir ruido tipo Poisson'''
    rng = np.random.default_rng()
    noise = rng.normal(0, snr_factor, I.size)
    return I + noise.reshape(I.shape)

# ========== FUNCIONES DE AJUSTE ==========

class AjustadorDifraccion:
    def __init__(self, M=512, lmda=600e-9, ua=40):
        self.M = M
        self.lmda = lmda
        self.ua = ua
        self.z = ua * 1.496e11
        self.plano = calc_plano(3000, lmda, ua)
        
    def elipse(self, a, b, theta=0):
        '''Crear apertura elíptica'''
        x = np.linspace(-self.plano/2, self.plano/2, self.M)
        X, Y = np.meshgrid(x, x)
        ct, st = np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)
        Xr = ct*X + st*Y
        Yr = -st*X + ct*Y
        U = np.ones((self.M, self.M))
        U[(Xr/a)**2 + (Yr/b)**2 <= 1] = 0
        return U
    
    def circular(self, d):
        '''Crear apertura circular'''
        x = np.linspace(-self.plano/2, self.plano/2, self.M)
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        U = np.ones((self.M, self.M))
        U[r <= d/2] = 0
        return U
    
    def cuadrado(self, d):
        '''Crear apertura cuadrada'''
        x = np.linspace(-self.plano/2, self.plano/2, self.M)
        X, Y = np.meshgrid(x, x)
        U = np.ones((self.M, self.M))
        U[(np.abs(X) <= d/2) & (np.abs(Y) <= d/2)] = 0
        return U
    
    def binario(self, d):
        '''Crear apertura binaria de contacto'''
        x = np.linspace(-self.plano/2, self.plano/2, self.M)
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        r1 = (d/2) * 0.65
        r2 = np.sqrt((d/2)**2 - r1**2)
        U = np.ones((self.M, self.M))
        U[(r >= r1) & (r <= d/2)] = 0
        U[(r >= r2) & (r <= d/2)] = 0
        return U
    
    def generar_perfil(self, forma, a, b, theta, impacto):
        '''Generar perfil de difracción completo'''
        if forma == 'elipse':
            pupila = self.elipse(a, b, theta)
        elif forma == 'circular':
            pupila = self.circular(a)
        elif forma == 'cuadrado':
            pupila = self.cuadrado(a)
        elif forma == 'binario':
            pupila = self.binario(a)
        else:
            pupila = self.circular(a)
        
        # Calcular difracción
        I = fresnel(pupila, self.M, self.plano, self.z, self.lmda)
        
        # Extraer perfil 1D
        x, y = extraer_perfil(I, self.M, self.plano, 0, impacto)
        
        # Normalizar
        y_norm = y / np.median(y[:50])
        
        return x, y_norm
    
    def chi_cuadrada(self, observado, calculado, error=0.03):
        '''Calcular chi-cuadrada'''
        chi2 = np.sum(((observado - calculado) / error)**2)
        return chi2
    
    def r_squared(self, observado, calculado):
        '''Calcular R-cuadrado'''
        ss_res = np.sum((observado - calculado)**2)
        ss_tot = np.sum((observado - np.mean(observado))**2)
        return 1 - (ss_res / ss_tot)

# ========== INTERFAZ INTERACTIVA ==========

def interfaz_interactiva():
    '''Crear interfaz interactiva con sliders'''
    
    # Inicializar
    ajustador = AjustadorDifraccion(M=512, lmda=600e-9, ua=40)
    
    # Generar datos sintéticos
    x_ref, y_ref = ajustador.generar_perfil('elipse', 1800, 1100, 0.5, 300)
    rng = np.random.default_rng(42)
    y_obs = y_ref + rng.normal(0, 0.03, y_ref.size)
    
    # Crear figura
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Ajuste Interactivo de Perfiles de Difracción', fontsize=16, fontweight='bold')
    
    # Subplot principal: perfiles
    ax_perfil = plt.subplot(2, 2, (1, 3))
    ax_perfil.set_xlabel('Posición [km]')
    ax_perfil.set_ylabel('Flujo Normalizado')
    ax_perfil.grid(True, alpha=0.3)
    
    line_obs, = ax_perfil.plot(x_ref/1000, y_obs, 'o', label='Observado', markersize=4, alpha=0.7)
    line_cal, = ax_perfil.plot(x_ref/1000, y_ref, '-', label='Calculado', linewidth=2)
    
    # Subplot de residuos
    ax_residual = plt.subplot(2, 2, 4)
    ax_residual.set_xlabel('Posición [km]')
    ax_residual.set_ylabel('Residuo')
    ax_residual.grid(True, alpha=0.3)
    line_res, = ax_residual.plot(x_ref/1000, np.zeros_like(y_ref), 'r-', linewidth=1)
    
    # Sliders
    ax_forma = plt.axes([0.2, 0.85, 0.15, 0.04])
    rax_forma = plt.axes([0.2, 0.70, 0.15, 0.12])
    
    ax_a = plt.axes([0.55, 0.85, 0.15, 0.04])
    sl_a = Slider(ax_a, 'a [m]', 1000, 3000, valinit=1800, valstep=100)
    
    ax_b = plt.axes([0.55, 0.80, 0.15, 0.04])
    sl_b = Slider(ax_b, 'b [m]', 500, 2000, valinit=1100, valstep=100)
    
    ax_theta = plt.axes([0.55, 0.75, 0.15, 0.04])
    sl_theta = Slider(ax_theta, 'θ [°]', 0, 360, valinit=0.5, valstep=1)
    
    ax_impact = plt.axes([0.55, 0.70, 0.15, 0.04])
    sl_impact = Slider(ax_impact, 'Impacto [m]', -1000, 1000, valinit=300, valstep=50)
    
    ax_ruido = plt.axes([0.55, 0.65, 0.15, 0.04])
    sl_ruido = Slider(ax_ruido, 'SNR factor', 0.001, 0.1, valinit=0.03, valstep=0.005)
    
    # Radio buttons para forma
    radio_forma = RadioButtons(rax_forma, ('elipse', 'circular', 'cuadrado', 'binario'), 
                               active=0)
    
    # Texto de información
    ax_info = plt.axes([0.2, 0.52, 0.5, 0.12])
    ax_info.axis('off')
    text_info = ax_info.text(0.05, 0.9, '', fontsize=10, family='monospace',
                             transform=ax_info.transAxes, verticalalignment='top')
    
    def actualizar(val=None):
        forma = radio_forma.value_selected
        a = sl_a.val
        b = sl_b.val
        theta = sl_theta.val
        impacto = sl_impact.val
        ruido = sl_ruido.val
        
        try:
            # Generar perfil calculado
            x_cal, y_cal = ajustador.generar_perfil(forma, a, b, theta, impacto)
            
            # Añadir ruido a los datos observados
            y_obs_ruido = y_obs + rng.normal(0, ruido, y_obs.size)
            
            # Calcular métricas
            chi2 = ajustador.chi_cuadrada(y_obs_ruido, y_cal, error=ruido)
            r2 = ajustador.r_squared(y_obs_ruido, y_cal)
            dof = len(y_obs) - 4
            chi2_red = chi2 / dof if dof > 0 else chi2
            
            # Actualizar gráficos
            line_obs.set_ydata(y_obs_ruido)
            line_cal.set_ydata(y_cal)
            residuos = y_obs_ruido - y_cal
            line_res.set_ydata(residuos)
            ax_residual.set_ylim([residuos.min()*1.2, residuos.max()*1.2])
            
            # Actualizar información
            info_text = f'''
PARÁMETROS:
  Forma: {forma}
  a = {a:.0f} m  |  b = {b:.0f} m
  θ = {theta:.1f}°  |  Impacto = {impacto:.0f} m
  SNR factor = {ruido:.4f}

MÉTRICAS DE AJUSTE:
  χ² = {chi2:.2f}
  χ²ᵣₑd = {chi2_red:.3f}
  R² = {r2:.4f}

DEGENERACIONES:
  ⚠️ Cambios en forma, tamaño e impacto
     pueden compensarse mutuamente
  ⚠️ El ajuste puede no ser único
'''
            text_info.set_text(info_text)
            
        except Exception as e:
            text_info.set_text(f'Error: {str(e)}')
        
        fig.canvas.draw_idle()
    
    # Conectar eventos
    sl_a.on_changed(actualizar)
    sl_b.on_changed(actualizar)
    sl_theta.on_changed(actualizar)
    sl_impact.on_changed(actualizar)
    sl_ruido.on_changed(actualizar)
    radio_forma.on_clicked(lambda label: actualizar())
    
    actualizar()
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.show()

# ========== BÚSQUEDA DE DEGENERACIONES ==========

def buscar_degeneraciones(y_obs, ajustador, n_intentos=100):
    '''Buscar múltiples conjuntos de parámetros que ajusten bien los datos'''
    
    resultados = []
    
    # Grid de búsqueda grueso
    for a in np.linspace(1400, 2200, 5):
        for b in np.linspace(800, 1400, 5):
            for theta in np.linspace(0, 5, 3):
                for impacto in np.linspace(-500, 500, 5):
                    try:
                        x, y_cal = ajustador.generar_perfil('elipse', a, b, theta, impacto)
                        chi2 = ajustador.chi_cuadrada(y_obs, y_cal, error=0.03)
                        r2 = ajustador.r_squared(y_obs, y_cal)
                        
                        resultados.append({
                            'a': a,
                            'b': b,
                            'theta': theta,
                            'impacto': impacto,
                            'chi2': chi2,
                            'r2': r2,
                            'y_cal': y_cal
                        })
                    except:
                        pass
    
    # Ordenar por chi2
    resultados = sorted(resultados, key=lambda x: x['chi2'])
    
    return resultados[:20]  # Retornar los 20 mejores

# ========== FUNCIONES DE ANÁLISIS ==========

def analizar_degeneraciones(y_obs, ajustador):
    '''Análisis detallado de degeneraciones'''
    
    print("="*70)
    print("BÚSQUEDA DE DEGENERACIONES EN EL ESPACIO DE PARÁMETROS")
    print("="*70)
    
    degeneraciones = buscar_degeneraciones(y_obs, ajustador)
    
    print(f"\nTop 10 mejores ajustes (χ² más bajo):\n")
    print(f"{'Rank':<5} {'χ²':<10} {'R²':<8} {'a [m]':<8} {'b [m]':<8} {'θ [°]':<8} {'Impacto [m]':<12}")
    print("-"*70)
    
    for i, deg in enumerate(degeneraciones[:10]):
        print(f"{i+1:<5} {deg['chi2']:<10.2f} {deg['r2']:<8.4f} {deg['a']:<8.0f} "
              f"{deg['b']:<8.0f} {deg['theta']:<8.1f} {deg['impacto']:<12.0f}")
    
    print("\n" + "="*70)
    print("INTERPRETACIÓN:")
    print("="*70)
    print("""
✓ Si múltiples conjuntos de parámetros (a, b, θ, impacto) producen χ² similar,
  existe DEGENERACIÓN: el problema es degenerado y hay múltiples soluciones.

✓ Patrón típico: cambios compensatorios
  - Aumentar 'a' se compensa con disminuir 'b'
  - Cambiar 'impacto' simula cambio en 'θ'
  - Variar 'θ' puede compensarse con 'impacto'

⚠️ Esto es el PROBLEMA CENTRAL en ocultaciones estelares:
  es imposible determinar únicamente forma, tamaño, orientación e impacto
  a partir de un único perfil de difracción.
""")
    
    return degeneraciones

if __name__ == "__main__":
    print("\n🔭 Herramienta de Ajuste de Perfiles de Difracción\n")
    print("Opciones:")
    print("1. Interfaz Interactiva (sliders en tiempo real)")
    print("2. Análisis de Degeneraciones (búsqueda exhaustiva)")
    print("3. Salir\n")
    
    opcion = input("Selecciona una opción (1-3): ").strip()
    
    if opcion == "1":
        print("\nLanzando interfaz interactiva...")
        interfaz_interactiva()
    
    elif opcion == "2":
        print("\nGenerando datos sintéticos...")
        ajustador = AjustadorDifraccion(M=512, lmda=600e-9, ua=40)
        x_ref, y_ref = ajustador.generar_perfil('elipse', 1800, 1100, 0.5, 300)
        rng = np.random.default_rng(42)
        y_obs = y_ref + rng.normal(0, 0.03, y_ref.size)
        
        print("Buscando degeneraciones...")
        degeneraciones = analizar_degeneraciones(y_obs, ajustador)
        
        # Graficar top 4 degeneraciones
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Degeneraciones: Múltiples Ajustes Válidos', fontsize=14, fontweight='bold')
        
        for idx, (ax, deg) in enumerate(zip(axes.flat, degeneraciones[:4])):
            ax.plot(x_ref/1000, y_obs, 'ko', label='Observado', markersize=4, alpha=0.6)
            ax.plot(x_ref/1000, deg['y_cal'], 'b-', label='Calculado', linewidth=2)
            ax.set_xlabel('Posición [km]')
            ax.set_ylabel('Flujo')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"χ²={deg['chi2']:.2f} | a={deg['a']:.0f} b={deg['b']:.0f} "
                        f"θ={deg['theta']:.1f}° imp={deg['impacto']:.0f}m")
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/degeneraciones.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Gráfico guardado en /mnt/user-data/outputs/degeneraciones.png")
    
    else:
        print("Saliendo...")
