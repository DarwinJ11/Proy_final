''''
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, symbols, diff, N, lambdify
from PIL import Image, ImageTk
import numpy as np

class RootFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Métodos Numéricos para Raíces")
        
        try:
            self.logo_img = Image.open("logo.png")
            self.logo_img = self.logo_img.resize((80, 80), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(self.logo_img)
            
            logo_frame = tk.Frame(self.root)
            logo_frame.pack(pady=5)
            
            tk.Label(logo_frame, image=self.logo).pack(side=tk.LEFT, padx=10)
            tk.Label(logo_frame, 
                    text="Instituto Tecnológico de Tuxtla Gutierrez\nMÉTODOS NUMÉRICOS PARA RAÍCES",
                    font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        except:
            tk.Label(self.root, 
                    text="Instituto Tecnológico de Tuxtla Gutierrez\nMÉTODOS NUMÉRICOS PARA RAÍCES",
                    font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        tk.Label(control_frame, text="Método:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.method_var = tk.StringVar(value="Bisección")
        self.method_menu = ttk.Combobox(control_frame, textvariable=self.method_var, 
                                        values=["Bisección", "Falsa Posición", "Newton-Raphson", "Secante", "Gauss-Seidel"])
        self.method_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.method_var.trace("w", self.toggle_entries)

        tk.Label(control_frame, text="Función f(x):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.entry_func = tk.Entry(control_frame, width=25)
        self.entry_func.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.entry_func.insert(0, "x**3 - 2*x - 5") 

        self.label_a = tk.Label(control_frame, text="Extremo izquierdo (a):")
        self.entry_a = tk.Entry(control_frame, width=15)
        self.entry_a.insert(0, "2.0")

        self.label_b = tk.Label(control_frame, text="Extremo derecho (b):")
        self.entry_b = tk.Entry(control_frame, width=15)
        self.entry_b.insert(0, "3.0")

        self.label_x0 = tk.Label(control_frame, text="Valor inicial (x0):")
        self.entry_x0 = tk.Entry(control_frame, width=15)
        self.entry_x0.insert(0, "2.5")

        self.label_x1 = tk.Label(control_frame, text="Segundo valor (x1):")
        self.entry_x1 = tk.Entry(control_frame, width=15)
        self.entry_x1.insert(0, "2.7")

        tk.Label(control_frame, text="Tolerancia:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.entry_tol = tk.Entry(control_frame, width=15)
        self.entry_tol.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.entry_tol.insert(0, "1e-6")

        tk.Label(control_frame, text="Máx iteraciones:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.entry_max_iter = tk.Entry(control_frame, width=15)
        self.entry_max_iter.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.entry_max_iter.insert(0, "100")

        tk.Label(control_frame, text="Decimales a mostrar:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.entry_decimals = tk.Entry(control_frame, width=15)
        self.entry_decimals.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.entry_decimals.insert(0, "6")

        tk.Button(control_frame, text="Calcular", command=self.compute, bg="#4CAF50", fg="white")\
            .grid(row=7, column=0, columnspan=2, pady=10)

        self.label_a.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_a.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.label_b.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.entry_b.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.label_x1.grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.entry_x1.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        results_frame = tk.Frame(main_frame)
        results_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.columns = ("Iteración", "x", "f(x)")
        self.table = ttk.Treeview(results_frame, columns=self.columns, show="headings", height=10)
        for col in self.columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=100)
        self.table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scrollbar.set)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        self.toggle_entries()

    def toggle_entries(self, *args):
        method = self.method_var.get()
        self.label_a.grid_remove()
        self.entry_a.grid_remove()
        self.label_b.grid_remove()
        self.entry_b.grid_remove()
        self.label_x0.grid_remove()
        self.entry_x0.grid_remove()
        self.label_x1.grid_remove()
        self.entry_x1.grid_remove()

        if method in ["Bisección", "Falsa Posición"]:
            self.label_a.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_a.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            self.label_b.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            self.entry_b.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        elif method == "Newton-Raphson":
            self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        elif method in ["Secante", "Gauss-Seidel"]:
            self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            self.label_x1.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            self.entry_x1.grid(row=3, column=1, padx=5, pady=5, sticky="w")

    def is_diagonally_dominant(self, A):
        """Verifica si la matriz es diagonal dominante"""
        D = np.diag(np.abs(A))
        S = np.sum(np.abs(A), axis=1) - D
        return np.all(D > S)

    def gauss_seidel_method(self, tol, max_iter):
        try:
            # Obtener los valores iniciales del usuario
            x0 = float(self.entry_x0.get())
            x1 = float(self.entry_x1.get())
            
            # Definir un sistema de ejemplo (3x3)
            A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 3]], dtype=float)
            b = np.array([15, 10, 10], dtype=float)
            
            # Verificar convergencia
            if not self.is_diagonally_dominant(A):
                messagebox.showwarning("Advertencia", "La matriz no es diagonal dominante. La convergencia no está garantizada.")
            
            # Inicializar el vector solución con los valores proporcionados por el usuario
            x = np.array([x0, x1, (x0+x1)/2])[:len(b)]  # Ajustar según dimensión del sistema
            
            results = []

            for i in range(max_iter):
                x_new = np.copy(x)
                for j in range(len(b)):
                    s1 = np.dot(A[j, :j], x_new[:j])
                    s2 = np.dot(A[j, j+1:], x[j+1:])
                    x_new[j] = (b[j] - s1 - s2) / A[j, j]
                
                error = np.linalg.norm(x_new - x, ord=np.inf)
                results.append((i, x_new.copy(), error))
                
                if error < tol:
                    break
                x = x_new
            
            return results, x
        
        except Exception as e:
            messagebox.showerror("Error", f"Error en Gauss-Seidel:\n{str(e)}")
            return [], None

    def bisection_method(self, f, a, b, tol, max_iter):
        x = symbols('x')
        f_expr = sympify(f)
        f_lambda = lambda val: N(f_expr.subs(x, val))
        iterations = []
        if f_lambda(a) * f_lambda(b) >= 0:
            messagebox.showerror("Error", "El intervalo [a, b] no cumple con f(a)*f(b) < 0.")
            return [], None
        for i in range(max_iter):
            c = (a + b) / 2
            f_c = f_lambda(c)
            iterations.append((i, c, f_c))
            if abs(f_c) < tol: break
            if f_lambda(a) * f_c < 0: b = c
            else: a = c
        return iterations, c

    def false_position_method(self, f, a, b, tol, max_iter):
        x = symbols('x')
        f_expr = sympify(f)
        f_lambda = lambda val: N(f_expr.subs(x, val))
        iterations = []
        if f_lambda(a) * f_lambda(b) >= 0:
            messagebox.showerror("Error", "El intervalo [a, b] no cumple con f(a)*f(b) < 0.")
            return [], None
        for i in range(max_iter):
            c = (a * f_lambda(b) - b * f_lambda(a)) / (f_lambda(b) - f_lambda(a))
            f_c = f_lambda(c)
            iterations.append((i, c, f_c))
            if abs(f_c) < tol: break
            if f_lambda(a) * f_c < 0: b = c
            else: a = c
        return iterations, c

    def newton_raphson_method(self, f, x0, tol, max_iter):
        x = symbols('x')
        f_expr = sympify(f)
        f_lambda = lambda val: N(f_expr.subs(x, val))
        df_expr = diff(f_expr, x)
        df_lambda = lambda val: N(df_expr.subs(x, val))
        iterations = []
        xn = x0
        for i in range(max_iter):
            fxn = f_lambda(xn)
            dfxn = df_lambda(xn)
            if dfxn == 0:
                messagebox.showerror("Error", "Derivada cero. No se puede continuar.")
                return [], None
            xn1 = xn - fxn / dfxn
            fxn1 = f_lambda(xn1)
            iterations.append((i, xn1, fxn1))
            if abs(fxn1) < tol: break
            xn = xn1
        return iterations, xn1

    def secant_method(self, f, x0, x1, tol, max_iter):
        x = symbols('x')
        f_lambda = lambdify(x, sympify(f), "numpy")
        iterations = []
        for i in range(max_iter):
            f0, f1 = f_lambda(x0), f_lambda(x1)
            if f1 - f0 == 0:
                messagebox.showerror("Error", "División por cero detectada. El método falla.")
                return [], None
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = f_lambda(x2)
            iterations.append((i, x2, f2))
            if abs(x2 - x1) < tol:
                break
            x0, x1 = x1, x2
        return iterations, x2

    def compute(self):
        try:
            method = self.method_var.get()
            f = self.entry_func.get()
            tol = float(self.entry_tol.get())
            max_iter = int(self.entry_max_iter.get())
            decimals = int(self.entry_decimals.get())

            if method in ["Bisección", "Falsa Posición"]:
                a = float(self.entry_a.get())
                b = float(self.entry_b.get())
                results, root = (self.bisection_method if method == "Bisección" else self.false_position_method)(f, a, b, tol, max_iter)
            elif method == "Newton-Raphson":
                x0 = float(self.entry_x0.get())
                results, root = self.newton_raphson_method(f, x0, tol, max_iter)
            elif method == "Secante":
                x0 = float(self.entry_x0.get())
                x1 = float(self.entry_x1.get())
                results, root = self.secant_method(f, x0, x1, tol, max_iter)
            elif method == "Gauss-Seidel":
                results, root = self.gauss_seidel_method(tol, max_iter)

            # Limpiar tabla
            for row in self.table.get_children():
                self.table.delete(row)

            if results:
                self.ax.clear()
                x_vals, y_vals = [], []
                for i, x_val, fx in results:
                    if isinstance(x_val, np.ndarray):
                        display_x = ', '.join(f"{val:.{decimals}f}" for val in x_val)
                        self.table.insert("", "end", values=(i, display_x, f"{fx:.{decimals}e}"))
                        y_vals.append(fx)
                    else:
                        self.table.insert("", "end", values=(i, f"{x_val:.{decimals}f}", f"{fx:.{decimals}e}"))
                        y_vals.append(abs(fx))
                    x_vals.append(i)

                self.ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='royalblue')
                self.ax.set_xlabel("Iteración")
                self.ax.set_ylabel("|f(x)|" if method != "Gauss-Seidel" else "Error")
                self.ax.set_title(f"Convergencia del Método de {method}")
                self.ax.grid(True, linestyle='--', alpha=0.7)
                self.canvas.draw()

                if root is not None:
                    display_root = root if not isinstance(root, np.ndarray) else ', '.join(f"{r:.{decimals}f}" for r in root)
                    messagebox.showinfo("Resultado", f"Solución aproximada:\n{display_root}\nIteraciones: {len(results)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error en los datos de entrada:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RootFinderApp(root)
    root.mainloop()
'''
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, symbols, diff, N, lambdify, expand, simplify
from PIL import Image, ImageTk
import numpy as np

class RootFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Métodos Numéricos")
        
        try:
            self.logo_img = Image.open("logo.png")
            self.logo_img = self.logo_img.resize((80, 80), Image.LANCZOS)
            self.logo = ImageTk.PhotoImage(self.logo_img)
            
            logo_frame = tk.Frame(self.root)
            logo_frame.pack(pady=5)
            
            tk.Label(logo_frame, image=self.logo).pack(side=tk.LEFT, padx=10)
            tk.Label(logo_frame, 
                    text="Instituto Tecnológico de Tuxtla Gutierrez\nMÉTODOS NUMÉRICOS",
                    font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        except:
            tk.Label(self.root, 
                    text="Instituto Tecnológico de Tuxtla Gutierrez\nMÉTODOS NUMÉRICOS",
                    font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Configuración de los widgets de la interfaz
        tk.Label(control_frame, text="Método:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.method_var = tk.StringVar(value="Bisección")
        self.method_menu = ttk.Combobox(control_frame, textvariable=self.method_var, 
                                      values=["Bisección", "Falsa Posición", "Newton-Raphson", 
                                              "Secante", "Gauss-Seidel", "Interpolación Lagrange",
                                              "Runge-Kutta 2do Orden", "Runge-Kutta 4to Orden"])
        self.method_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.method_var.trace("w", self.toggle_entries)

        self.label_func = tk.Label(control_frame, text="Función f(x):")
        self.entry_func = tk.Entry(control_frame, width=25)
        self.entry_func.insert(0, "x**3 - 2*x - 5")

        self.label_a = tk.Label(control_frame, text="Extremo izquierdo (a):")
        self.entry_a = tk.Entry(control_frame, width=15)
        self.entry_a.insert(0, "2.0")

        self.label_b = tk.Label(control_frame, text="Extremo derecho (b):")
        self.entry_b = tk.Entry(control_frame, width=15)
        self.entry_b.insert(0, "3.0")

        self.label_x0 = tk.Label(control_frame, text="Valor inicial (x0):")
        self.entry_x0 = tk.Entry(control_frame, width=15)
        self.entry_x0.insert(0, "2.5")

        self.label_x1 = tk.Label(control_frame, text="Segundo valor (x1):")
        self.entry_x1 = tk.Entry(control_frame, width=15)
        self.entry_x1.insert(0, "2.7")

        self.label_y0 = tk.Label(control_frame, text="Valor inicial (y0):")
        self.entry_y0 = tk.Entry(control_frame, width=15)
        self.entry_y0.insert(0, "1.0")

        self.label_h = tk.Label(control_frame, text="Tamaño paso (h):")
        self.entry_h = tk.Entry(control_frame, width=15)
        self.entry_h.insert(0, "0.1")

        self.label_xn = tk.Label(control_frame, text="Valor final (xn):")
        self.entry_xn = tk.Entry(control_frame, width=15)
        self.entry_xn.insert(0, "1.0")

        tk.Label(control_frame, text="Tolerancia:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.entry_tol = tk.Entry(control_frame, width=15)
        self.entry_tol.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.entry_tol.insert(0, "1e-6")

        tk.Label(control_frame, text="Máx iteraciones:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.entry_max_iter = tk.Entry(control_frame, width=15)
        self.entry_max_iter.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.entry_max_iter.insert(0, "100")

        tk.Label(control_frame, text="Decimales a mostrar:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.entry_decimals = tk.Entry(control_frame, width=15)
        self.entry_decimals.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.entry_decimals.insert(0, "6")

        tk.Button(control_frame, text="Calcular", command=self.compute, bg="#4CAF50", fg="white")\
            .grid(row=7, column=0, columnspan=2, pady=10)

        # Botón para ingresar puntos de interpolación
        self.btn_interp_points = tk.Button(control_frame, text="Ingresar puntos", 
                                         command=self.ingresar_puntos_interpolacion,
                                         state=tk.DISABLED)
        self.btn_interp_points.grid(row=8, column=0, columnspan=2, pady=5)

        results_frame = tk.Frame(main_frame)
        results_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.columns = ("Iteración", "x", "y")
        self.table = ttk.Treeview(results_frame, columns=self.columns, show="headings", height=10)
        for col in self.columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=100)
        self.table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scrollbar.set)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Variables para interpolación
        self.x_vals = []
        self.y_vals = []
        self.x_interp = None
        
        self.toggle_entries()

    def toggle_entries(self, *args):
        method = self.method_var.get()
        # Ocultar todos los campos primero
        self.label_a.grid_remove()
        self.entry_a.grid_remove()
        self.label_b.grid_remove()
        self.entry_b.grid_remove()
        self.label_x0.grid_remove()
        self.entry_x0.grid_remove()
        self.label_x1.grid_remove()
        self.entry_x1.grid_remove()
        self.label_func.grid_remove()
        self.entry_func.grid_remove()
        self.label_y0.grid_remove()
        self.entry_y0.grid_remove()
        self.label_h.grid_remove()
        self.entry_h.grid_remove()
        self.label_xn.grid_remove()
        self.entry_xn.grid_remove()
        self.btn_interp_points.config(state=tk.DISABLED)

        if method == "Interpolación Lagrange":
            self.btn_interp_points.config(state=tk.NORMAL)
            return

        # Mostrar función para métodos que lo requieren
        if method not in ["Gauss-Seidel", "Runge-Kutta 2do Orden", "Runge-Kutta 4to Orden"]:
            self.label_func.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            self.entry_func.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        if method in ["Bisección", "Falsa Posición"]:
            self.label_a.config(text="Extremo izquierdo (a):")
            self.label_a.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_a.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            
            self.label_b.config(text="Extremo derecho (b):")
            self.label_b.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            self.entry_b.grid(row=3, column=1, padx=5, pady=5, sticky="w")
            
        elif method == "Newton-Raphson":
            self.label_x0.config(text="Valor inicial (x0):")
            self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            
        elif method in ["Secante", "Gauss-Seidel"]:
            self.label_x0.config(text="Primer valor (x0):" if method == "Secante" else "Valor inicial x1:")
            self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            
            self.label_x1.config(text="Segundo valor (x1):" if method == "Secante" else "Valor inicial x2:")
            self.label_x1.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            self.entry_x1.grid(row=3, column=1, padx=5, pady=5, sticky="w")
            
        elif method in ["Runge-Kutta 2do Orden", "Runge-Kutta 4to Orden"]:
            self.label_func.config(text="Función f(x,y):")
            self.label_func.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            self.entry_func.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            self.entry_func.delete(0, tk.END)
            self.entry_func.insert(0, "x - y")  # Ejemplo para EDO
            
            self.label_x0.config(text="Valor inicial x0:")
            self.label_x0.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            self.entry_x0.grid(row=2, column=1, padx=5, pady=5, sticky="w")
            
            self.label_y0.config(text="Valor inicial y0:")
            self.label_y0.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            self.entry_y0.grid(row=3, column=1, padx=5, pady=5, sticky="w")
            
            self.label_h.config(text="Tamaño paso (h):")
            self.label_h.grid(row=4, column=0, padx=5, pady=5, sticky="e")
            self.entry_h.grid(row=4, column=1, padx=5, pady=5, sticky="w")
            
            self.label_xn.config(text="Valor final (xn):")
            self.label_xn.grid(row=5, column=0, padx=5, pady=5, sticky="e")
            self.entry_xn.grid(row=5, column=1, padx=5, pady=5, sticky="w")

    def ingresar_puntos_interpolacion(self):
        # Ventana para ingresar número de puntos
        n = simpledialog.askinteger("Puntos de interpolación", 
                                   "Ingrese el número de puntos:", 
                                   parent=self.root, minvalue=2, maxvalue=10)
        if n is None:
            return
            
        self.x_vals = []
        self.y_vals = []
        
        # Ventana para ingresar cada punto
        for i in range(n):
            while True:
                punto = simpledialog.askstring(f"Punto {i+1}", 
                                             f"Ingrese el punto {i+1} (x,y):",
                                             parent=self.root)
                if punto is None:
                    return
                    
                try:
                    x, y = map(float, punto.split(','))
                    self.x_vals.append(x)
                    self.y_vals.append(y)
                    break
                except:
                    messagebox.showerror("Error", "Formato incorrecto. Use 'x,y' (ej. 2.5,3.7)")
        
        # Pedir valor a interpolar
        while True:
            x_interp = simpledialog.askstring("Valor a interpolar", 
                                            "Ingrese el valor de x a interpolar:",
                                            parent=self.root)
            if x_interp is None:
                return
                
            try:
                self.x_interp = float(x_interp)
                break
            except:
                messagebox.showerror("Error", "Ingrese un número válido")

    def lagrange_interpolacion(self):
        try:
            x = symbols('x')
            n = len(self.x_vals)
            polinomio = 0

            for i in range(n):
                L_i = 1
                for j in range(n):
                    if i != j:
                        L_i *= (x - self.x_vals[j]) / (self.x_vals[i] - self.x_vals[j])
                polinomio += self.y_vals[i] * L_i

            polinomio_simplificado = simplify(expand(polinomio))
            f_interp = lambdify(x, polinomio_simplificado, 'numpy')
            y_interp = f_interp(self.x_interp)

            # Mostrar resultados
            result_text = f"Polinomio de interpolación de Lagrange:\n{polinomio_simplificado}"
            result_text += f"\n\nValor interpolado en x = {self.x_interp}: {y_interp:.6f}"
            
            # Graficar
            self.ax.clear()
            x_plot = np.linspace(min(self.x_vals)-1, max(self.x_vals)+1, 500)
            y_plot = f_interp(x_plot)

            self.ax.plot(x_plot, y_plot, label='Polinomio Interpolante', color='blue')
            self.ax.plot(self.x_vals, self.y_vals, 'ro', label='Puntos Dados')
            self.ax.plot(self.x_interp, y_interp, 'ks', label='Punto Interpolado')
            self.ax.set_title('Interpolación de Lagrange')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('f(x)')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            # Limpiar tabla y mostrar polinomio
            for row in self.table.get_children():
                self.table.delete(row)
                
            self.table.insert("", "end", values=("Polinomio:", str(polinomio_simplificado), ""))
            self.table.insert("", "end", values=("Interpolación:", f"x = {self.x_interp}", f"y = {y_interp:.6f}"))
            
            messagebox.showinfo("Resultados", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en interpolación:\n{str(e)}")

    def runge_kutta_2(self, f, x0, y0, xn, h):
        xs = [x0]
        ys = [y0]
        x, y = x0, y0
        
        # Limpiar tabla
        for row in self.table.get_children():
            self.table.delete(row)
            
        # Agregar condición inicial
        self.table.insert("", "end", values=(0, f"{x:.6f}", f"{y:.6f}"))
        
        i = 1
        while x < xn:
            k1 = h * f(x, y)
            k2 = h * f(x + h, y + k1)
            y += 0.5 * (k1 + k2)
            x += h
            xs.append(x)
            ys.append(y)
            
            # Agregar a la tabla
            self.table.insert("", "end", values=(i, f"{x:.6f}", f"{y:.6f}"))
            i += 1
            
            # Verificar si hemos alcanzado xn
            if x >= xn:
                break
                
        return xs, ys

    def runge_kutta_4(self, f, x0, y0, xn, h):
        xs = [x0]
        ys = [y0]
        x, y = x0, y0
        
        # Limpiar tabla
        for row in self.table.get_children():
            self.table.delete(row)
            
        # Agregar condición inicial
        self.table.insert("", "end", values=(0, f"{x:.6f}", f"{y:.6f}"))
        
        i = 1
        while x < xn:
            k1 = h * f(x, y)
            k2 = h * f(x + h/2, y + k1/2)
            k3 = h * f(x + h/2, y + k2/2)
            k4 = h * f(x + h, y + k3)
            y += (k1 + 2*k2 + 2*k3 + k4) / 6
            x += h
            xs.append(x)
            ys.append(y)
            
            # Agregar a la tabla
            self.table.insert("", "end", values=(i, f"{x:.6f}", f"{y:.6f}"))
            i += 1
            
            # Verificar si hemos alcanzado xn
            if x >= xn:
                break
                
        return xs, ys

    def is_diagonally_dominant(self, A):
        """Verifica si la matriz es diagonal dominante"""
        D = np.diag(np.abs(A))
        S = np.sum(np.abs(A), axis=1) - D
        return np.all(D > S)

    def gauss_seidel_method(self, tol, max_iter):
        try:
            # Obtener valores iniciales
            x0 = float(self.entry_x0.get())
            x1 = float(self.entry_x1.get())
            
            # Sistema de ecuaciones lineal de ejemplo (3x3)
            A = np.array([[4, -1, 0], 
                         [-1, 4, -1], 
                         [0, -1, 3]], dtype=float)
            b = np.array([15, 10, 10], dtype=float)
            
            # Verificar convergencia
            if not self.is_diagonally_dominant(A):
                messagebox.showwarning("Advertencia", 
                                     "La matriz no es diagonal dominante. La convergencia no está garantizada.")
            
            # Inicializar vector solución
            x = np.array([x0, x1, (x0+x1)/2])[:len(b)]
            results = []
            
            for i in range(max_iter):
                x_new = np.copy(x)
                for j in range(len(b)):
                    s1 = np.dot(A[j, :j], x_new[:j])
                    s2 = np.dot(A[j, j+1:], x[j+1:])
                    x_new[j] = (b[j] - s1 - s2) / A[j, j]
                
                error = np.linalg.norm(x_new - x, ord=np.inf)
                results.append((i, x_new.copy(), error))
                
                if error < tol:
                    break
                    
                x = x_new
            
            return results, x
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el método de Gauss-Seidel:\n{str(e)}")
            return [], None

    def bisection_method(self, f, a, b, tol, max_iter):
        try:
            x = symbols('x')
            f_expr = sympify(f)
            f_lambda = lambda val: float(N(f_expr.subs(x, val)))
            iterations = []
            
            # Validación inicial
            if a >= b:
                messagebox.showerror("Error", "El extremo izquierdo (a) debe ser menor que el extremo derecho (b).")
                return [], None
                
            fa, fb = f_lambda(a), f_lambda(b)
            if fa * fb >= 0:
                messagebox.showerror("Error", "El intervalo [a, b] no cumple con f(a)*f(b) < 0.")
                return [], None
                
            for i in range(max_iter):
                c = (a + b) / 2
                fc = f_lambda(c)
                iterations.append((i, c, fc))
                
                if abs(fc) < tol:
                    break
                    
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
            return iterations, c
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el método de Bisección:\n{str(e)}")
            return [], None

    def false_position_method(self, f, a, b, tol, max_iter):
        try:
            x = symbols('x')
            f_expr = sympify(f)
            f_lambda = lambda val: float(N(f_expr.subs(x, val)))
            iterations = []
            
            # Validación inicial
            if a >= b:
                messagebox.showerror("Error", "El extremo izquierdo (a) debe ser menor que el extremo derecho (b).")
                return [], None
                
            fa, fb = f_lambda(a), f_lambda(b)
            if fa * fb >= 0:
                messagebox.showerror("Error", "El intervalo [a, b] no cumple con f(a)*f(b) < 0.")
                return [], None
                
            for i in range(max_iter):
                # Evitar división por cero
                if fb - fa == 0:
                    messagebox.showerror("Error", "División por cero detectada. El método falla.")
                    return [], None
                    
                c = (a * fb - b * fa) / (fb - fa)
                fc = f_lambda(c)
                iterations.append((i, c, fc))
                
                if abs(fc) < tol:
                    break
                    
                if fa * fc < 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
                    
            return iterations, c
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el método de Falsa Posición:\n{str(e)}")
            return [], None

    def newton_raphson_method(self, f, x0, tol, max_iter):
        try:
            x = symbols('x')
            f_expr = sympify(f)
            f_lambda = lambda val: float(N(f_expr.subs(x, val)))
            df_expr = diff(f_expr, x)
            df_lambda = lambda val: float(N(df_expr.subs(x, val)))
            iterations = []
            
            xn = float(x0)
            for i in range(max_iter):
                fxn = f_lambda(xn)
                dfxn = df_lambda(xn)
                
                if dfxn == 0:
                    messagebox.showerror("Error", "Derivada cero. No se puede continuar.")
                    return [], None
                    
                xn1 = xn - fxn / dfxn
                fxn1 = f_lambda(xn1)
                iterations.append((i, xn1, fxn1))
                
                if abs(fxn1) < tol:
                    break
                    
                xn = xn1
                
            return iterations, xn1
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el método de Newton-Raphson:\n{str(e)}")
            return [], None

    def secant_method(self, f, x0, x1, tol, max_iter):
        try:
            x = symbols('x')
            f_expr = sympify(f)
            f_lambda = lambda val: float(N(f_expr.subs(x, val)))
            iterations = []
            
            xn0, xn1 = float(x0), float(x1)
            fxn0, fxn1 = f_lambda(xn0), f_lambda(xn1)
            
            for i in range(max_iter):
                if fxn1 - fxn0 == 0:
                    messagebox.showerror("Error", "División por cero detectada. El método falla.")
                    return [], None
                    
                xn2 = xn1 - fxn1 * (xn1 - xn0) / (fxn1 - fxn0)
                fxn2 = f_lambda(xn2)
                iterations.append((i, xn2, fxn2))
                
                if abs(xn2 - xn1) < tol:
                    break
                    
                xn0, xn1 = xn1, xn2
                fxn0, fxn1 = fxn1, fxn2
                
            return iterations, xn2
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el método de la Secante:\n{str(e)}")
            return [], None

    def compute(self):
        try:
            method = self.method_var.get()
            tol = float(self.entry_tol.get())
            max_iter = int(self.entry_max_iter.get())
            decimals = int(self.entry_decimals.get())

            if method == "Interpolación Lagrange":
                if len(self.x_vals) < 2:
                    messagebox.showerror("Error", "Debe ingresar al menos 2 puntos para interpolación.")
                    return
                if self.x_interp is None:
                    messagebox.showerror("Error", "Debe ingresar un valor de x a interpolar.")
                    return
                
                self.lagrange_interpolacion()
                return

            if method in ["Runge-Kutta 2do Orden", "Runge-Kutta 4to Orden"]:
                f_expr = self.entry_func.get()
                x0 = float(self.entry_x0.get())
                y0 = float(self.entry_y0.get())
                xn = float(self.entry_xn.get())
                h = float(self.entry_h.get())
                
                # Crear función para Runge-Kutta
                x_sym, y_sym = symbols('x y')
                f = lambdify((x_sym, y_sym), sympify(f_expr), 'numpy')
                
                if method == "Runge-Kutta 2do Orden":
                    xs, ys = self.runge_kutta_2(f, x0, y0, xn, h)
                else:
                    xs, ys = self.runge_kutta_4(f, x0, y0, xn, h)
                
                # Graficar resultados
                self.ax.clear()
                self.ax.plot(xs, ys, marker='o', linestyle='-', color='royalblue')
                self.ax.set_title(f"Solución con {method}")
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')
                self.ax.grid(True)
                self.canvas.draw()
                
                return

            if method == "Gauss-Seidel":
                results, root = self.gauss_seidel_method(tol, max_iter)
            else:
                f = self.entry_func.get()
                if not f.strip():
                    messagebox.showerror("Error", "Debe ingresar una función.")
                    return
                    
                if method in ["Bisección", "Falsa Posición"]:
                    a = float(self.entry_a.get())
                    b = float(self.entry_b.get())
                    results, root = (self.bisection_method if method == "Bisección" 
                                    else self.false_position_method)(f, a, b, tol, max_iter)
                elif method == "Newton-Raphson":
                    x0 = float(self.entry_x0.get())
                    results, root = self.newton_raphson_method(f, x0, tol, max_iter)
                elif method == "Secante":
                    x0 = float(self.entry_x0.get())
                    x1 = float(self.entry_x1.get())
                    results, root = self.secant_method(f, x0, x1, tol, max_iter)

            self.show_results(method, results, root, decimals)
            
        except ValueError as ve:
            messagebox.showerror("Error", f"Datos de entrada inválidos:\n{str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado:\n{str(e)}")

    def show_results(self, method, results, root, decimals):
        # Limpiar tabla
        for row in self.table.get_children():
            self.table.delete(row)
        
        if results:
            self.ax.clear()
            x_vals, y_vals = [], []
            
            for i, x_val, fx in results:
                if isinstance(x_val, np.ndarray):
                    display_x = ', '.join(f"{val:.{decimals}f}" for val in x_val)
                    self.table.insert("", "end", values=(i, display_x, f"{fx:.{decimals}e}"))
                    y_vals.append(fx)
                else:
                    self.table.insert("", "end", values=(i, f"{x_val:.{decimals}f}", f"{fx:.{decimals}e}"))
                    y_vals.append(abs(fx))
                x_vals.append(i)

            self.ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='royalblue')
            self.ax.set_xlabel("Iteración")
            self.ax.set_ylabel("|f(x)|" if method != "Gauss-Seidel" else "Error")
            self.ax.set_title(f"Convergencia del Método de {method}")
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas.draw()

            if root is not None:
                display_root = root if not isinstance(root, np.ndarray) else ', '.join(f"{r:.{decimals}f}" for r in root)
                messagebox.showinfo("Resultado", f"Solución aproximada:\n{display_root}\nIteraciones: {len(results)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RootFinderApp(root)
    root.mainloop()