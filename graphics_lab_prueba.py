from tkinter import filedialog as fl
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

import matplotlib
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('TkAgg')

# --- PARCHE DE COMPATIBILIDAD PARA NUMPY MODERNO ---
np.int = int
np.float = float
# ---------------------------------------------------

import editor_lab
import plots_lab
from tkinter import Frame
from scipy.interpolate import interp1d

class Aplicacion:

    def __init__(self):
        self.ventana1 = tk.Tk()
        self.ventana1.protocol('WM_DELETE_WINDOW', self.ventana1.quit)
        self.ventana1.title('SW-CAA-Shapes-LAB')
        self.ventana1.configure(bg='white')
        
        # --- AJUSTE DE RESOLUCIÓN ---
        self.ventana1.geometry('1280x720')
        self.fullScreenState = False
        self.ventana1.attributes('-fullscreen', False)
        
        self.ventana1.bind("<F11>", lambda event: self.ventana1.attributes("-fullscreen", not self.ventana1.attributes("-fullscreen")))
        self.ventana1.bind("<Escape>", lambda event: self.ventana1.attributes("-fullscreen", False))
        
        # Variables de instancia
        self.secs = None
        self.normflx = None
        self.img = None
        self.I1f = None
        self.D = None
        
        # --- CONFIGURACIÓN RESPONSIVA DEL GRID PRINCIPAL ---
        self.ventana1.grid_columnconfigure(0, weight=1)
        self.ventana1.grid_columnconfigure(1, weight=2)
        
        self.ventana1.grid_rowconfigure(0, weight=1) 
        self.ventana1.grid_rowconfigure(1, weight=0) 
        self.ventana1.grid_rowconfigure(2, weight=0)
        
        self.canvas1 = editor_lab.Editor(self.ventana1, canvas_size=320)
        self.canvas1.grid(column=0, row=0, pady=5) 
        
        e, e2, e3, e4, fs = plots_lab.plot()
        self.f = fs
        
        self.f.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08, wspace=0.3, hspace=0.5)
        
        self.p = e
        self.linea1c1, = self.p.plot([], [], '-', color='red', label='Perfil real')
        self.linea2c1, = self.p.plot([], [], '-', color='blue', label='Perfil muestreo')
        self.p.set_xlabel("Distance [m]", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 10})
        self.p.set_ylabel("Flux", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 10})
        self.p.set_title("Diffraction profile", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
        
        self.p2 = e2
        self.linea1c2, = self.p2.plot([], [], '-', color='red', label='Data')
        self.linea2c2, = self.p2.plot([], [], '-', color='blue', label='Simulated')
        self.p2.set_xlabel("Time [s]", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 10})
        self.p2.set_title("Sampled profile", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})

        self.c2 = e3
        self.c2.set_xlabel("FOV", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
        self.c4 = e4
        self.c4.set_xlabel("Diff Pattern", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})

        self.canvas = FigureCanvasTkAgg(self.f, self.ventana1)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=2, sticky='nsew', padx=10, pady=10)

        self.toolbarFrame = Frame(master=self.ventana1, bg='white')
        self.toolbarFrame.grid(row=2, column=1, sticky='w', padx=10)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        
        self.prints()
        self.params()
        self.ventana1.mainloop()

    def prints(self):
        self.M2 = tk.StringVar(value="1024")
        self.vE2 = tk.StringVar(value="5e-4") 
        self.vr2 = tk.StringVar(value="0")
        self.ang2 = tk.StringVar(value="10e-9")
        self.fps2 = tk.StringVar(value="20")
        self.mV2 = tk.StringVar(value="14")
        self.nEst2 = tk.StringVar(value="0")
        self.nLamb2 = tk.StringVar(value="1")
        self.ua2 = tk.StringVar(value="2.18")
        self.d2 = tk.StringVar(value="2e-3")
        self.T2 = tk.StringVar(value="0")
        self.b2 = tk.StringVar(value="0")
        self.toffset2 = tk.StringVar(value="0")
        self.cent2 = tk.StringVar(value="212")
        self.expt2 = tk.StringVar(value="0.05")
        self.lmda2 = tk.StringVar(value="650e-9")
        self.chi2 = tk.StringVar(value="999.0")
        self.skiprows2 = tk.StringVar(value="14")
        
        self.auto_b_var = tk.BooleanVar(value=True) 
        
    def params(self):
        self.my_frame = tk.Frame(self.ventana1, bg="cyan")
        self.my_frame.grid(column=0, row=1, sticky='s', padx=5, pady=(0, 10))
        
        # 2D Calculation parameters
        tk.Label(self.my_frame, text="2D CALC PARAMETERS", bg="cyan", font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=2, pady=(2,2))
        tk.Label(self.my_frame, text="Mesh size:", bg="cyan").grid(row=1, column=0, sticky='e')
        tk.Entry(self.my_frame, textvariable=self.M2, width=8).grid(row=1, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Distance [m]:", bg="cyan").grid(row=2, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.ua2).grid(row=2, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Object diam [m]:", bg="cyan").grid(row=3, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.d2).grid(row=3, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Nº Wlengths:", bg="cyan").grid(row=4, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.nLamb2).grid(row=4, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Apparent mag:", bg="cyan").grid(row=5, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.mV2).grid(row=5, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Source Size [m]:", bg="cyan").grid(row=6, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.nEst2).grid(row=6, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Sep Wlengths [m]:", bg="cyan").grid(row=7, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.ang2).grid(row=7, column=1, padx=2, pady=1)
        tk.Label(self.my_frame, text="Wavelength [m]:", bg="cyan").grid(row=8, column=0, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.lmda2).grid(row=8, column=1, padx=2, pady=1)

        # 1D Calculation parameters
        tk.Label(self.my_frame, text="1D CALC PARAMETERS", bg="cyan", font=("Arial", 9, "bold")).grid(row=0, column=2, columnspan=2, pady=(2,2))
        tk.Label(self.my_frame, text="Cam speed [m/s]:", bg="cyan").grid(row=1, column=2, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.vE2).grid(row=1, column=3, padx=2, pady=1)
        tk.Label(self.my_frame, text="Object speed [m/s]:", bg="cyan").grid(row=2, column=2, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.vr2).grid(row=2, column=3, padx=2, pady=1)
        tk.Label(self.my_frame, text="Time Offset [s]:", bg="cyan").grid(row=3, column=2, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.toffset2).grid(row=3, column=3, padx=2, pady=1)
        tk.Label(self.my_frame, text="Fps:", bg="cyan").grid(row=4, column=2, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.fps2).grid(row=4, column=3, padx=2, pady=1)
        tk.Label(self.my_frame, text="Reading Dir [deg]:", bg="cyan").grid(row=5, column=2, sticky='e')
        tk.Entry(self.my_frame, width=8, textvariable=self.T2).grid(row=5, column=3, padx=2, pady=1)
        
        tk.Label(self.my_frame, text="Impact param [m]:", bg="cyan").grid(row=6, column=2, sticky='e')
        b_frame = tk.Frame(self.my_frame, bg="cyan")
        b_frame.grid(row=6, column=3, sticky='w')
        tk.Entry(b_frame, width=6, textvariable=self.b2).pack(side=tk.LEFT)
        tk.Checkbutton(b_frame, text="Auto", variable=self.auto_b_var, bg="cyan", font=("Arial", 7)).pack(side=tk.LEFT)
        
        tk.Label(self.my_frame, text="CHi2 :", bg="cyan").grid(row=7, column=2, sticky='e')
        tk.Entry(self.my_frame, width=12, textvariable=self.chi2, bg='red', fg='white').grid(row=7, column=3, padx=2, pady=1)
        
        # Data Parameters tangra
        tk.Label(self.my_frame, text="TANGRA PARMS:", bg="cyan", font=("Arial", 9, "bold")).grid(row=8, column=2, sticky='e', pady=1)
        
        tg_frame = tk.Frame(self.my_frame, bg="cyan")
        tg_frame.grid(row=8, column=3, sticky='w')
        tk.Label(tg_frame, text="Cent:", bg="cyan").pack(side=tk.LEFT)
        tk.Entry(tg_frame, width=4, textvariable=self.cent2).pack(side=tk.LEFT, padx=1)
        tk.Label(tg_frame, text="Exp[s]:", bg="cyan").pack(side=tk.LEFT)
        tk.Entry(tg_frame, width=4, textvariable=self.expt2).pack(side=tk.LEFT)
        
        # Action Buttons
        btn_frame = tk.Frame(self.my_frame, bg="cyan")
        btn_frame.grid(row=9, column=0, columnspan=4, pady=10)
        
        tk.Button(btn_frame, text="2D Calc", command=self.get_pattern, width=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="1D Calc", command=self.get_curves, width=8).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Load CSV", command=self.cargar_datos, width=8).pack(side=tk.LEFT, padx=4)
        
        tk_skip = tk.Frame(btn_frame, bg="cyan")
        tk_skip.pack(side=tk.LEFT, padx=4)
        tk.Label(tk_skip, text="Skiprows:", bg="cyan", font=("Arial", 8)).pack(side=tk.LEFT)
        tk.Entry(tk_skip, width=4, textvariable=self.skiprows2, bg='green', fg='white').pack(side=tk.LEFT)
        
        tk.Button(btn_frame, text="Reset", command=self.canvas1.create_poly, width=8).pack(side=tk.LEFT, padx=4)

    def get_pattern(self):
        self.img, self.I1f, self.D = self.canvas1.process_image(
            M=int(self.M2.get()), mV=float(self.mV2.get()), nEst=float(self.nEst2.get()), 
            nLamb=int(self.nLamb2.get()), d=float(self.d2.get()), ua=float(self.ua2.get()), 
            lmda=float(self.lmda2.get()), ang=float(self.ang2.get())
        )
        self.draw_image(self.img, self.I1f, self.D, b=float(self.b2.get()), T=float(self.T2.get()))

    def calc_chi2_for_b(self, b_guess):
        try:
            t_off = float(self.toffset2.get())
            
            x, y, x1, y1 = self.canvas1.process_curves(
                self.I1f, self.D, M=int(self.M2.get()), vE=float(self.vE2.get()), 
                vr=float(self.vr2.get()), fps=float(self.fps2.get()), mV=float(self.mV2.get()), 
                ua=float(self.ua2.get()), 
                toffset=0.0,
                T=float(self.T2.get()), b=b_guess
            )
            
            x1 = np.array(x1) + t_off
            
            sort_idx = np.argsort(x1)
            x1_sorted = x1[sort_idx]
            y1_sorted = y1[sort_idx]
            x1_unq, unq_idx = np.unique(x1_sorted, return_index=True)
            
            if len(x1_unq) > 1:
                func = interp1d(x1_unq, y1_sorted[unq_idx], bounds_error=False, fill_value=1e-6) 
                sim = func(self.secs)
                sim_safe = np.clip(sim, 1e-6, None)
                chi_stat = np.sum((self.normflx - sim_safe)**2 / sim_safe)
                return chi_stat
            else:
                return 1e12 
        except Exception:
            return 1e12

    def get_curves(self):
        if self.img is None or self.I1f is None:
            print("Try Calcutating 2D pattern FIRST!!")
            return
            
        print("Reading 2D Pattern")
        
        # --- BÚSQUEDA OPTIMIZADA: COARSE-TO-FINE ---
        if self.auto_b_var.get() and self.secs is not None and self.normflx is not None:
            print("Iniciando optimización del parámetro de impacto (Fase Gruesa)...")
            
            M = int(self.M2.get())
            pixel_size = self.D / M
            limit_b = self.D * 0.48 
            
            # FASE 1: Búsqueda Rápida (Saltos de 20 píxeles)
            coarse_step = pixel_size * 20
            b_tests_coarse = np.arange(-limit_b, limit_b, coarse_step)
            
            min_chi_coarse = np.inf
            best_b_coarse = 0.0
            
            for b_test in b_tests_coarse:
                chi_val = self.calc_chi2_for_b(b_test)
                if chi_val < min_chi_coarse:
                    min_chi_coarse = chi_val
                    best_b_coarse = b_test
                    
            print(f"Zona localizada en b aprox: {best_b_coarse:.4e}. Refinando...")
            
            # FASE 2: Refinamiento Fino (Píxel por píxel solo en la zona vecina)
            limit_low = best_b_coarse - coarse_step
            limit_high = best_b_coarse + coarse_step
            b_tests_fine = np.arange(limit_low, limit_high, pixel_size)
            
            min_chi_final = min_chi_coarse
            best_b_final = best_b_coarse
            
            for b_test in b_tests_fine:
                chi_val = self.calc_chi2_for_b(b_test)
                if chi_val < min_chi_final:
                    min_chi_final = chi_val
                    best_b_final = b_test
            
            self.b2.set(f"{best_b_final:.4e}")
            if min_chi_final != np.inf:
                self.chi2.set(f"{min_chi_final:.2f}")
                
            print(f"Mejor parámetro de impacto (b) final: {best_b_final:.4e} (Chi2: {min_chi_final:.2f})")
        # -------------------------------------------

        try:
            t_off = float(self.toffset2.get())
            x, y, x1, y1 = self.canvas1.process_curves(
                self.I1f, self.D, M=int(self.M2.get()), vE=float(self.vE2.get()), 
                vr=float(self.vr2.get()), fps=float(self.fps2.get()), mV=float(self.mV2.get()), 
                ua=float(self.ua2.get()), 
                toffset=0.0, 
                T=float(self.T2.get()), b=float(self.b2.get())
            )
            
            x1 = np.array(x1) + t_off
            self.draw_graphic(x, y, x1, y1, False)
            
        except IndexError as e:
            self.chi2.set("ERR: LÍMITES")
            self.draw_graphic([], [], [], [], False, error_mode=True)
            print(f"\n[ERROR DE TRAZADO]: La cámara excedió los bordes.")
        except Exception as e:
            self.chi2.set("ERR: SIM 1D")
            self.draw_graphic([], [], [], [], False, error_mode=True)
            print(f"Error inesperado en simulación 1D: {e}")

    def draw_graphic(self, x, y, x1, y1, band, error_mode=False):
        if not error_mode:
            if band:
               self.linea1c1.set_data(x, y) 
               self.linea1c2.set_data(x1, y1)
            elif len(x) > 0:
                self.linea2c1.set_data(x, y)
                self.linea2c2.set_data(x1, y1)
        
        if self.secs is not None and self.normflx is not None and len(x1) > 0 and not error_mode:
            try:  
                sort_idx = np.argsort(x1)
                x1_sorted = x1[sort_idx]
                y1_sorted = y1[sort_idx]
                
                x1_unq, unq_idx = np.unique(x1_sorted, return_index=True)
                
                if len(x1_unq) > 1:
                    func = interp1d(x1_unq, y1_sorted[unq_idx], bounds_error=False, fill_value=1e-6) 
                    sim = func(self.secs)
                    sim_safe = np.clip(sim, 1e-6, None)
                    chi_stat = np.sum((self.normflx - sim_safe)**2 / sim_safe)
                    self.chi2.set(round(chi_stat, 4))
                else:
                    self.chi2.set("ERR: INFO 1D")
            except Exception as e:
                print(f"Error calculando Chi2: {e}")
                self.chi2.set("ERR: INTERP")
        
        if self.D is not None:
            M = int(self.M2.get())
            x_arr = np.linspace(-self.D/2, self.D/2, M)
            T = float(self.T2.get())
            b = float(self.b2.get())
            
            x_plot1 = x_arr * np.cos(T*np.pi/180) - b * np.sin(T*np.pi/180)
            x_plot2 = x_arr * np.sin(T*np.pi/180) + b * np.cos(T*np.pi/180)
            
            self.c4.clear()
            self.c4.set_xlabel("Diff Pattern", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
            if self.I1f is not None:
                self.c4.imshow(self.I1f, extent=[-self.D/2, self.D/2, -self.D/2, self.D/2], cmap=plt.cm.gray)
            self.c4.plot(x_plot1, x_plot2, 'r-')
            self.c4.set_xlim([-self.D/2, self.D/2])
            self.c4.set_ylim([-self.D/2, self.D/2])
            
            self.c2.clear()
            self.c2.set_xlabel("FOV", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
            if self.img is not None:
                self.c2.imshow(self.img, extent=[-self.D/2, self.D/2, -self.D/2, self.D/2], cmap=plt.cm.gray)
            self.c2.plot(x_plot1, x_plot2, 'r-')
            self.c2.set_xlim([-self.D/2, self.D/2])
            self.c2.set_ylim([-self.D/2, self.D/2])

        self.p2.relim()
        self.p2.autoscale_view()
        self.p.relim()
        self.p.autoscale_view()
        self.f.canvas.draw()
        
    def draw_image(self, img, I1f, D, b, T):
        self.c4.clear()
        self.c4.set_xlabel("Diff Pattern", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
        self.c4.imshow(I1f, extent=[-D/2, D/2, -D/2, D/2], cmap=plt.cm.gray)
        self.c4.set_xlim([-D/2, D/2])
        self.c4.set_ylim([-D/2, D/2])
        
        self.c2.clear()
        self.c2.set_xlabel("FOV", fontdict={'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12})
        self.c2.imshow(img, extent=[-D/2, D/2, -D/2, D/2], cmap=plt.cm.gray)
        self.c2.set_xlim([-D/2, D/2])
        self.c2.set_ylim([-D/2, D/2])
        
        self.f.canvas.draw()

    def cargar_datos(self):
        file = fl.askopenfilename()
        if file != '':
            try:
                kk = np.loadtxt(file, delimiter=',', skiprows=int(self.skiprows2.get()))
                flx = kk[:, 2]
                self.normflx = flx / (np.mean(flx[-20:])) 
                
                exp = float(self.expt2.get())
                cent = int(self.cent2.get())
                num = kk[:, 0]
                num = num - cent 
                self.secs = exp * num
                
                self.draw_graphic([], [], self.secs, self.normflx, True)
            except Exception as e:
                print(f"Error cargando archivo: {e}")

if __name__ == "__main__":
    aplicacion1 = Aplicacion()