# app/gui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os

from app.calculations import mkr_solver, dqm_solver, plate_data, comparison_solver
from app.postprocessing import plotter
from app.export import word_exporter

PLOT_CONFIG = {

    'displacements': {'title': 'Поле прогибов w', 'cbar': 'Прогиб w, мм', 'multiplier': 1000,
                      'cmap': 'hsv', 'levels': 64},
    'Mx': {'title': 'Поле моментов Mx', 'cbar': 'Mx, кН·м/м', 'multiplier': 1 / 1000,
           'cmap': 'hsv'},
    'My': {'title': 'Поле моментов My', 'cbar': 'My, кН·м/м', 'multiplier': 1 / 1000,
           'cmap': 'hsv'},
    'Mxy': {'title': 'Поле моментов Mxy', 'cbar': 'Mxy, кН·м/м', 'multiplier': 1 / 1000,
            'cmap': 'hsv'},
    'Qx': {'title': 'Поле поперечных сил Qx', 'cbar': 'Qx, кН/м', 'multiplier': 1 / 1000,
           'cmap': 'hsv'},
    'Qy': {'title': 'Поле поперечных сил Qy', 'cbar': 'Qy, кН/м', 'multiplier': 1 / 1000,
           'cmap': 'hsv'},
    'sigma_x': {'title': 'Поле напряжений σ_x', 'cbar': 'σ_x, МПа', 'multiplier': 1 / 1e6,
                'cmap': 'hsv'},
    'sigma_y': {'title': 'Поле напряжений σ_y', 'cbar': 'σ_y, МПа', 'multiplier': 1 / 1e6,
                'cmap': 'hsv'},
    'tau_xy': {'title': 'Поле напряжений τ_xy', 'cbar': 'τ_xy, МПа', 'multiplier': 1 / 1e6,
               'cmap': 'hsv'},

    'def_center_conv': {'title': 'Сходимость: Прогиб в центре', 'ylabel': 'Прогиб w, мм',
                        'multiplier': 1000, 'xlabel': 'Сетка (МКР интервалы / МДК узлы)'},
    'mx_max_conv': {'title': 'Сходимость: Макс. |Mx|', 'ylabel': 'Макс. |Mx|, Н·м/м',
                    'multiplier': 1, 'xlabel': 'Сетка'},
    'my_max_conv': {'title': 'Сходимость: Макс. |My|', 'ylabel': 'Макс. |My|, Н·м/м',
                    'multiplier': 1, 'xlabel': 'Сетка'},
    'mxy_max_conv': {'title': 'Сходимость: Макс. |Mxy|', 'ylabel': 'Макс. |Mxy|, Н·м/м',
                     'multiplier': 1, 'xlabel': 'Сетка'},
    'qx_max_conv': {'title': 'Сходимость: Макс. |Qx|', 'ylabel': 'Макс. |Qx|, Н/м', 'multiplier': 1,
                    'xlabel': 'Сетка'},
    'qy_max_conv': {'title': 'Сходимость: Макс. |Qy|', 'ylabel': 'Макс. |Qy|, Н/м', 'multiplier': 1,
                    'xlabel': 'Сетка'},
    'sig_x_max_conv': {'title': 'Сходимость: Макс. |σx|', 'ylabel': 'Макс. |σx|, Па',
                       'multiplier': 1, 'xlabel': 'Сетка'},
    'sig_y_max_conv': {'title': 'Сходимость: Макс. |σy|', 'ylabel': 'Макс. |σy|, Па',
                       'multiplier': 1, 'xlabel': 'Сетка'},
    'tau_xy_max_conv': {'title': 'Сходимость: Макс. |τxy|', 'ylabel': 'Макс. |τxy|, Па',
                        'multiplier': 1, 'xlabel': 'Сетка'},

    'time_conv': {'title': 'Сходимость: Время расчета', 'ylabel': 'Время, с', 'multiplier': 1,
                  'xlabel': 'Сетка'},
}
DEFAULT_PLOT_KEYS_ORDER = [
    'displacements', 'Mx', 'My', 'Mxy', 'Qx', 'Qy',
    'sigma_x', 'sigma_y', 'tau_xy'
]
CONVERGENCE_PLOT_KEYS_ORDER = [

    'def_center_conv', 'mx_max_conv', 'my_max_conv', 'mxy_max_conv',
    'qx_max_conv', 'qy_max_conv', 'sig_x_max_conv', 'sig_y_max_conv',
    'tau_xy_max_conv',
    'time_conv'
]


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Расчет изгиба плиты Софи-Жермен Лагранжа")
        self.geometry("1200x950")

        self.plate_type_var = tk.StringVar(value="стандартная")
        self.boundary_condition_var = tk.StringVar(value="шарнир")

        self.mkr_var = tk.BooleanVar(value=True)
        self.dqm_var = tk.BooleanVar()
        self.compare_methods_var = tk.BooleanVar()
        self.bubnov_galerkin_var = tk.BooleanVar()
        self.save_to_word_var = tk.BooleanVar()

        self.custom_length_var = tk.StringVar(value="6.0")
        self.custom_width_var = tk.StringVar(value="3.0")

        self.custom_thickness_var = tk.StringVar(value="0.12")
        self.custom_young_modulus_var = tk.StringVar(value="34.5e9")
        self.custom_poisson_ratio_var = tk.StringVar(value="0.2")
        self.custom_load_intensity_var = tk.StringVar(value="8000")

        # --- Переменные для выбора сеток ---
        self.PRESET_GRIDS = {
            "6x4": (6, 4),
            "12x8": (12, 8),
            "30x20": (30, 20),
            "Другая": None  # Для пользовательского ввода
        }
        self.mkr_grid_choice_var = tk.StringVar(value="30x20")  # По умолчанию
        self.mkr_custom_nx_var = tk.StringVar(value="30")
        self.mkr_custom_ny_var = tk.StringVar(value="20")

        self.dqm_grid_choice_var = tk.StringVar(value="30x20")  # По умолчанию
        self.dqm_custom_nx_var = tk.StringVar(value="30")
        self.dqm_custom_ny_var = tk.StringVar(value="20")

        self.dqm_grid_type_var = tk.StringVar(value="равномерная")  # Равномерная/Чебышева

        self.calculation_results = None
        self.current_plot_data_keys = []
        self.current_plot_index = 0
        self.is_comparison_plot = False  # Флаг, что отображается график сравнения

        self._create_main_layout()
        self._create_input_widgets()
        self._create_output_widgets()
        self._update_all_dependent_widgets()
        self._clear_plot_and_results()  # Начальная очистка

    def _update_all_dependent_widgets(self):
        self._toggle_custom_data_fields()
        self._update_bubnov_galerkin_availability()
        self._toggle_mkr_grid_options()
        self._toggle_dqm_options()
        self._update_plot_navigation_buttons_state()
        self._on_compare_methods_change()

    def _create_main_layout(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.control_frame = ttk.Frame(self, padding="10")
        self.control_frame.grid(row=0, column=0, sticky="nsew")

        self.output_frame = ttk.Frame(self, padding="10")
        self.output_frame.grid(row=0, column=1, sticky="nsew")

    def _create_input_widgets(self):
        current_row = 0
        # --- Группа "Тип пластины" ---
        plate_group = ttk.LabelFrame(self.control_frame, text="1. Тип пластины")
        plate_group.grid(row=current_row, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
        ttk.Radiobutton(plate_group, text="Стандартная пластина", variable=self.plate_type_var,
                        value="стандартная", command=self._on_plate_type_change).pack(anchor="w",
                                                                                      padx=5,
                                                                                      pady=(5, 0))
        ttk.Radiobutton(plate_group, text="Ввести свои данные", variable=self.plate_type_var,
                        value="пользовательская", command=self._on_plate_type_change).pack(
            anchor="w", padx=5, pady=(0, 5))
        self.custom_data_frame = ttk.Frame(plate_group)
        self.custom_data_frame.columnconfigure(1, weight=1)
        fields = [
            ("Длина (a), м:", self.custom_length_var), ("Ширина (b), м:", self.custom_width_var),
            ("Толщина (h), м:", self.custom_thickness_var),
            ("Модуль Юнга (E), Па:", self.custom_young_modulus_var),
            ("Коэф. Пуассона (ν):", self.custom_poisson_ratio_var),
            ("Нагрузка (q), Па:", self.custom_load_intensity_var)
        ]
        for i, (text, var) in enumerate(fields):
            ttk.Label(self.custom_data_frame, text=text).grid(row=i, column=0, sticky="w", padx=2,
                                                              pady=2)
            ttk.Entry(self.custom_data_frame, textvariable=var).grid(row=i, column=1, sticky="ew",
                                                                     padx=2, pady=2)
        current_row += 1

        # --- Группа "Вариант закрепления" ---
        boundary_group = ttk.LabelFrame(self.control_frame, text="2. Вариант закрепления")
        boundary_group.grid(row=current_row, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
        ttk.Radiobutton(boundary_group, text="Шарнирное опирание",
                        variable=self.boundary_condition_var, value="шарнир").pack(anchor="w",
                                                                                   padx=5)
        ttk.Radiobutton(boundary_group, text="Жесткая заделка",
                        variable=self.boundary_condition_var, value="заделка").pack(anchor="w",
                                                                                    padx=5)
        current_row += 1

        # --- Группа "Методы расчета и сетки" ---
        methods_group = ttk.LabelFrame(self.control_frame, text="3. Методы, опции и сетки")
        methods_group.grid(row=current_row, column=0, padx=5, pady=5, sticky="ew", columnspan=2)

        # --- МКР и его сетка ---
        self.mkr_checkbox = ttk.Checkbutton(methods_group, text="МКР", variable=self.mkr_var,
                                            command=self._toggle_mkr_grid_options)
        self.mkr_checkbox.pack(anchor="w", padx=5, pady=(5, 0))

        self.mkr_grid_frame = ttk.Frame(methods_group)

        ttk.Label(self.mkr_grid_frame, text="Сетка МКР (Nx × Ny):").pack(side=tk.LEFT, padx=(0, 5))
        self.mkr_grid_combobox = ttk.Combobox(self.mkr_grid_frame,
                                              textvariable=self.mkr_grid_choice_var,
                                              values=list(self.PRESET_GRIDS.keys()),
                                              state="readonly", width=10)
        self.mkr_grid_combobox.pack(side=tk.LEFT, padx=5)
        self.mkr_grid_combobox.bind("<<ComboboxSelected>>", self._on_mkr_grid_preset_change)

        self.mkr_custom_grid_frame = ttk.Frame(self.mkr_grid_frame)
        ttk.Label(self.mkr_custom_grid_frame, text="Nx:").pack(side=tk.LEFT)
        ttk.Entry(self.mkr_custom_grid_frame, textvariable=self.mkr_custom_nx_var, width=5).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.mkr_custom_grid_frame, text="Ny:").pack(side=tk.LEFT)
        ttk.Entry(self.mkr_custom_grid_frame, textvariable=self.mkr_custom_ny_var, width=5).pack(
            side=tk.LEFT)

        # --- МДК и его опции ---
        self.dqm_checkbox = ttk.Checkbutton(methods_group, text="МДК (DQM)", variable=self.dqm_var,
                                            command=self._toggle_dqm_options)
        self.dqm_checkbox.pack(anchor="w", padx=5, pady=(5, 0))

        self.dqm_options_frame = ttk.LabelFrame(methods_group, text="Опции МДК")

        # Тип сетки (равномерная/Чебышева)
        ttk.Label(self.dqm_options_frame, text="Тип узлов:").pack(anchor="w", padx=5, pady=(5, 0))
        ttk.Radiobutton(self.dqm_options_frame, text="Равномерные", variable=self.dqm_grid_type_var,
                        value="равномерная").pack(anchor="w", padx=15)
        ttk.Radiobutton(self.dqm_options_frame, text="Чебышева", variable=self.dqm_grid_type_var,
                        value="чебышева").pack(anchor="w", padx=15)

        # Выбор размера сетки для МДК
        ttk.Label(self.dqm_options_frame, text="Сетка МДК (Nx × Ny узлов):").pack(anchor="w",
                                                                                  padx=5,
                                                                                  pady=(5, 0))
        self.dqm_grid_combobox = ttk.Combobox(self.dqm_options_frame,
                                              textvariable=self.dqm_grid_choice_var,
                                              values=list(self.PRESET_GRIDS.keys()),
                                              state="readonly", width=10)
        self.dqm_grid_combobox.pack(anchor="w", padx=15, pady=(0, 5))
        self.dqm_grid_combobox.bind("<<ComboboxSelected>>", self._on_dqm_grid_preset_change)

        self.dqm_custom_grid_frame = ttk.Frame(self.dqm_options_frame)
        ttk.Label(self.dqm_custom_grid_frame, text="Nx (узлов):").pack(side=tk.LEFT)
        ttk.Entry(self.dqm_custom_grid_frame, textvariable=self.dqm_custom_nx_var, width=5).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.dqm_custom_grid_frame, text="Ny (узлов):").pack(side=tk.LEFT)
        ttk.Entry(self.dqm_custom_grid_frame, textvariable=self.dqm_custom_ny_var, width=5).pack(
            side=tk.LEFT)

        # --- Общие опции методов ---
        self.compare_methods_checkbox = ttk.Checkbutton(methods_group, text="Сравнение методов",
                                                        variable=self.compare_methods_var,
                                                        command=self._on_compare_methods_change)
        self.compare_methods_checkbox.pack(anchor="w", padx=5, pady=5)

        self.bubnov_checkbox = ttk.Checkbutton(methods_group,
                                               text="Значения Бубнова-Галеркина (станд.)",
                                               variable=self.bubnov_galerkin_var,
                                               command=self._on_bubnov_galerkin_change)
        self.bubnov_checkbox.pack(anchor="w", padx=5)
        current_row += 1

        # --- Группа "Экспорт" ---
        export_group = ttk.LabelFrame(self.control_frame, text="4. Экспорт")
        export_group.grid(row=current_row, column=0, padx=5, pady=5, sticky="ew", columnspan=2)
        ttk.Checkbutton(export_group, text="Сохранить графики в Word",
                        variable=self.save_to_word_var).pack(anchor="w", padx=5)
        current_row += 1

        calculate_button = ttk.Button(self.control_frame, text="Рассчитать",
                                      command=self._on_calculate)
        calculate_button.grid(row=current_row, column=0, columnspan=2, padx=5, pady=20)
        current_row += 1
        self.control_frame.grid_rowconfigure(current_row, weight=1)

        # Начальное обновление состояний
        self._update_all_dependent_widgets()

    def _on_mkr_grid_preset_change(self, event=None):
        choice = self.mkr_grid_choice_var.get()
        if choice == "Другая":
            self.mkr_custom_grid_frame.pack(side=tk.LEFT, padx=5, after=self.mkr_grid_combobox)
        else:
            self.mkr_custom_grid_frame.pack_forget()
            nx, ny = self.PRESET_GRIDS[choice]
            self.mkr_custom_nx_var.set(str(nx))
            self.mkr_custom_ny_var.set(str(ny))

    def _toggle_mkr_grid_options(self, event=None):
        if self.mkr_var.get():
            self.mkr_grid_frame.pack(anchor="w", padx=15, pady=(0, 10), fill="x",
                                     before=self.dqm_checkbox)
            self._on_mkr_grid_preset_change()
        else:
            self.mkr_grid_frame.pack_forget()
            self.mkr_custom_grid_frame.pack_forget()

    def _on_dqm_grid_preset_change(self, event=None):
        choice = self.dqm_grid_choice_var.get()
        if choice == "Другая":
            self.dqm_custom_grid_frame.pack(anchor="w", padx=5, pady=(0, 5),
                                            after=self.dqm_grid_combobox)
        else:
            self.dqm_custom_grid_frame.pack_forget()
            nx, ny = self.PRESET_GRIDS[choice]
            self.dqm_custom_nx_var.set(str(nx + 1))
            self.dqm_custom_ny_var.set(str(ny + 1))

            self.dqm_custom_nx_var.set(str(nx))
            self.dqm_custom_ny_var.set(str(ny))

    def _toggle_dqm_options(self, event=None):
        if self.dqm_var.get():
            self.dqm_options_frame.pack(anchor="w", padx=15, pady=5, fill="x",
                                        before=self.compare_methods_checkbox)
            self._on_dqm_grid_preset_change()
        else:
            self.dqm_options_frame.pack_forget()
            self.dqm_custom_grid_frame.pack_forget()

    def _on_compare_methods_change(self, event=None):

        if self.compare_methods_var.get():
            self.mkr_checkbox.config(state="disabled")
            self.dqm_checkbox.config(state="disabled")

            if self.plate_type_var.get() == "стандартная":
                self.bubnov_checkbox.config(state="normal")
            else:
                self.bubnov_galerkin_var.set(False)
                self.bubnov_checkbox.config(state="disabled")
        else:
            self.mkr_checkbox.config(state="normal")
            self.dqm_checkbox.config(state="normal")
            self._update_bubnov_galerkin_availability()

    def _create_output_widgets(self):
        self.output_frame.grid_rowconfigure(0, weight=20)
        self.output_frame.grid_rowconfigure(1, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)

        plot_area_frame = ttk.Frame(self.output_frame)
        plot_area_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        plot_area_frame.grid_rowconfigure(0, weight=1)
        plot_area_frame.grid_columnconfigure(0, weight=1)

        plot_frame_labelframe = ttk.LabelFrame(plot_area_frame, text="График")
        plot_frame_labelframe.grid(row=0, column=0, sticky="nsew")
        plot_frame_labelframe.grid_rowconfigure(0, weight=1)
        plot_frame_labelframe.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(7, 7), dpi=100)
        self.plot_ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame_labelframe)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame_labelframe)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        nav_button_frame = ttk.Frame(plot_area_frame)
        nav_button_frame.grid(row=1, column=0, pady=5, sticky="ew")
        self.prev_plot_button = ttk.Button(nav_button_frame, text="<< Предыдущий",
                                           command=self._show_prev_plot)
        self.prev_plot_button.pack(side=tk.LEFT, padx=10, expand=True)
        self.next_plot_button = ttk.Button(nav_button_frame, text="Следующий >>",
                                           command=self._show_next_plot)
        self.next_plot_button.pack(side=tk.RIGHT, padx=10, expand=True)

        results_frame = ttk.LabelFrame(self.output_frame, text="Значения параметров")
        results_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.results_text_area = scrolledtext.ScrolledText(results_frame, height=10, wrap=tk.WORD,
                                                           state="disabled")
        self.results_text_area.pack(expand=True, fill="both", padx=5, pady=5)

    def _clear_plot_and_results(self):
        self.plot_ax.clear()
        if hasattr(self.figure, "_colorbar_axes_list"):
            for cax_info in self.figure._colorbar_axes_list:
                try:
                    if isinstance(cax_info, dict) and 'cax' in cax_info:
                        self.figure.delaxes(cax_info['cax'])
                    elif hasattr(cax_info, 'figure'):
                        self.figure.delaxes(cax_info)
                except (KeyError, AttributeError, ValueError):
                    pass
            self.figure._colorbar_axes_list = []

        self.plot_ax.text(0.5, 0.5, "Для отображения графика выполните расчет",
                          ha='center', va='center', transform=self.plot_ax.transAxes)
        self.canvas.draw_idle()
        self.results_text_area.config(state="normal")
        self.results_text_area.delete("1.0", tk.END)
        self.results_text_area.insert(tk.END, "Результаты расчета будут отображены здесь.")
        self.results_text_area.config(state="disabled")
        self.calculation_results = None
        self.current_plot_data_keys = []
        self.current_plot_index = 0
        self.is_comparison_plot = False
        self._update_plot_navigation_buttons_state()

    def _display_current_plot(self):
        if not self.calculation_results or not self.current_plot_data_keys:
            return

        key_to_plot = self.current_plot_data_keys[self.current_plot_index]
        plot_info = PLOT_CONFIG.get(key_to_plot)

        if not plot_info:
            print(f"Нет конфигурации для графика: {key_to_plot}")
            self.plot_ax.clear()
            self.plot_ax.text(0.5, 0.5, f"Конфигурация для '{key_to_plot}' отсутствует",
                              ha='center', va='center', transform=self.plot_ax.transAxes)
            self.canvas.draw_idle()
            self._update_plot_navigation_buttons_state()
            return

        if self.is_comparison_plot:  # Для графиков сравнения
            X_values = self.calculation_results.get('x_axis_data')
            Y_mkr = self.calculation_results.get(f'{key_to_plot}_mkr')
            Y_dqm = self.calculation_results.get(f'{key_to_plot}_dqm')
            bubnov_val = self.calculation_results.get(f'{key_to_plot}_bubnov')

            plotter.plot_convergence_on_ax(
                self.plot_ax, X_values,
                Y_mkr, Y_dqm,
                plot_info['title'], plot_info.get('xlabel', 'Размер сетки (N)'),
                plot_info['ylabel'],
                "МКР",
                "МДК (Равномерная)" if self.dqm_grid_type_var.get() == "равномерная" else "МДК (Чебышев)",
                bubnov_value=bubnov_val if self.bubnov_galerkin_var.get() and self.plate_type_var.get() == "стандартная" else None,
                multiplier=plot_info.get('multiplier', 1.0)
            )

        else:  # Обычный контурный график
            X = self.calculation_results.get('X')
            Y = self.calculation_results.get('Y')
            Z = self.calculation_results.get(key_to_plot)

            if Z is not None and X is not None and Y is not None:
                Z_scaled = Z * plot_info.get('multiplier', 1.0)
                plotter.plot_contour_on_ax(self.plot_ax, X, Y, Z_scaled,
                                           plot_info['title'], plot_info['cbar'],
                                           key_to_plot,
                                           cmap=plot_info.get('cmap', 'viridis'),
                                           levels=plot_info.get('levels', 50))
            else:
                self.plot_ax.clear()
                self.plot_ax.text(0.5, 0.5,
                                  f"Данные для '{plot_info.get('title', key_to_plot)}' отсутствуют",
                                  ha='center', va='center', transform=self.plot_ax.transAxes)
                self.canvas.draw_idle()

        self._update_plot_navigation_buttons_state()

    def _update_plot_navigation_buttons_state(self):
        if not hasattr(self, 'prev_plot_button'): return
        num_plots = len(self.current_plot_data_keys)
        if num_plots <= 1:
            self.prev_plot_button.config(state="disabled")
            self.next_plot_button.config(state="disabled")
        else:
            self.prev_plot_button.config(
                state="normal" if self.current_plot_index > 0 else "disabled")
            self.next_plot_button.config(
                state="normal" if self.current_plot_index < num_plots - 1 else "disabled")

    def _show_prev_plot(self):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            self._display_current_plot()

    def _show_next_plot(self):
        if self.current_plot_index < len(self.current_plot_data_keys) - 1:
            self.current_plot_index += 1
            self._display_current_plot()

    def _update_numerical_results(self, results, method_name="МКР"):
        if not results:
            return

        self.results_text_area.config(state="normal")
        self.results_text_area.delete("1.0", tk.END)
        text_output = f"--- Ключевые результаты ({method_name}) ---\n"
        text_output += f"Пластина: {results.get('name', self.plate_type_var.get())}\n"
        text_output += f"Закрепление: {self.boundary_condition_var.get().capitalize()}\n"

        if method_name == "МКР":
            text_output += f"Сетка МКР: {results.get('Nx')}x{results.get('Ny')} интервалов\n"
        elif method_name == "МДК":
            text_output += f"Сетка DQM: {results.get('Nx_intervals_actual', results.get('Nx_intervals'))}x{results.get('Ny_intervals_actual', results.get('Ny_intervals'))} интервалов (узлов {results.get('Nx_points_actual', results.get('Nx_intervals', 0) + 1)}x{results.get('Ny_points_actual', results.get('Ny_intervals', 0) + 1)})\n"
            text_output += f"Тип узлов DQM: {results.get('grid_type', 'N/A').capitalize()}\n"

        text_output += f"Цилиндрическая жесткость D: {results.get('D_flex', 0):.3e} Н·м\n\n"

        displ_data = results.get('displacements')
        if displ_data is not None:
            global_w_max = np.max(displ_data)
            global_w_min = np.min(displ_data)
            text_output += f"Глобальный макс. прогиб: {global_w_max:.4e} м ({global_w_max * 1000:.4f} мм)\n"
            text_output += f"Глобальный мин. прогиб: {global_w_min:.4e} м ({global_w_min * 1000:.4f} мм)\n"

            # Проверка типа и размерности displ_data
            if isinstance(displ_data, np.ndarray) and displ_data.ndim >= 2 and displ_data.shape[
                0] > 0 and displ_data.shape[1] > 0:
                center_idx_y = displ_data.shape[0] // 2
                center_idx_x = displ_data.shape[1] // 2
                center_w_val = displ_data[center_idx_y, center_idx_x]
                if np.isnan(center_w_val):
                    text_output += f"Прогиб в центре: Н/Д (NaN в центре)\n\n"
                else:
                    text_output += f"Прогиб в центре: {center_w_val:.4e} м ({center_w_val * 1000:.4f} мм)\n\n"
            else:
                text_output += f"Прогиб в центре: Н/Д (данные некорректны для определения центра)\n\n"

        for key, name in [('Mx', 'Mx'), ('My', 'My'), ('Mxy', 'Mxy'), ('Qx', 'Qx'), ('Qy', 'Qy'),
                          ('sigma_x', 'σ_x'), ('sigma_y', 'σ_y'), ('tau_xy', 'τ_xy')]:
            data = results.get(key)
            if data is not None:
                unit = "кН·м/м" if 'M' in key else ("кН/м" if 'Q' in key else "МПа")
                multiplier = 1 / 1000 if 'M' in key or 'Q' in key else 1 / 1e6
                max_val = np.nanmax(data)
                min_val = np.nanmin(data)
                text_output += f"Макс. {name}: {max_val * multiplier:.3f} {unit}, Мин. {name}: {min_val * multiplier:.3f} {unit}\n"

                abs_data = np.abs(data)
                abs_max_val = np.nanmax(abs_data)
                if np.isnan(abs_max_val):
                    text_output += f"Абс. макс. {name}: Н/Д (все значения NaN или отсутствуют)\n"
                else:
                    text_output += f"Абс. макс. {name}: {abs_max_val * multiplier:.3f} {unit}\n"

                if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 0 and \
                        data.shape[1] > 0:
                    center_idx_y_param = data.shape[0] // 2
                    center_idx_x_param = data.shape[1] // 2
                    center_val_param = data[center_idx_y_param, center_idx_x_param]

                    if np.isnan(center_val_param):
                        text_output += f"{name} в центре: Н/Д (NaN в центре)\n"
                    else:
                        text_output += f"{name} в центре: {center_val_param * multiplier:.3f} {unit}\n"
                else:
                    text_output += f"{name} в центре: Н/Д (данные не 2D массив или некорректные размеры)\n"

        self.results_text_area.insert(tk.END, text_output)
        self.results_text_area.config(state="disabled")

    def _on_plate_type_change(self):
        self._toggle_custom_data_fields()
        self._update_bubnov_galerkin_availability()
        self._clear_plot_and_results()

    def _toggle_custom_data_fields(self):
        if hasattr(self, 'custom_data_frame'):
            if self.plate_type_var.get() == "пользовательская":
                self.custom_data_frame.pack(anchor="w", padx=5, pady=(0, 10), fill="x")
            else:
                self.custom_data_frame.pack_forget()

    def _update_bubnov_galerkin_availability(self):
        if hasattr(self, 'bubnov_checkbox'):
            if self.plate_type_var.get() == "пользовательская":
                self.bubnov_galerkin_var.set(False)
                self.bubnov_checkbox.config(state="disabled")
            else:
                self.bubnov_checkbox.config(state="normal")

    def _on_bubnov_galerkin_change(self):
        pass

    def _on_calculate(self):
        plate_type = self.plate_type_var.get()
        boundary_condition = self.boundary_condition_var.get()
        use_mkr_calc = self.mkr_var.get()
        use_dqm_calc = self.dqm_var.get()
        compare_methods_flag = self.compare_methods_var.get()
        add_bubnov = self.bubnov_galerkin_var.get()
        save_to_word = self.save_to_word_var.get()

        if plate_type == "стандартная":
            current_plate_params = plate_data.STANDARD_PLATE_PARAMS.copy()
        else:
            try:
                custom_data = {
                    'length': float(self.custom_length_var.get()),
                    'width': float(self.custom_width_var.get()),
                    'thickness': float(self.custom_thickness_var.get()),
                    'young_modulus': float(self.custom_young_modulus_var.get()),
                    'poisson_ratio': float(self.custom_poisson_ratio_var.get()),
                    'load_intensity': float(self.custom_load_intensity_var.get())}
                if not (custom_data['length'] > 0 and custom_data['width'] > 0 and
                        custom_data['thickness'] > 0 and custom_data['young_modulus'] > 0 and
                        -1 < custom_data['poisson_ratio'] < 0.5):
                    messagebox.showerror("Ошибка данных", "Параметры пластины некорректны.")
                    return
                current_plate_params = custom_data
            except ValueError:
                messagebox.showerror("Ошибка ввода",
                                     "Введите корректные числовые значения для параметров пластины.")
                return
        current_plate_params['name'] = current_plate_params.get('name', plate_type.capitalize())

        if plate_type == "пользовательская" and add_bubnov:
            messagebox.showwarning("Предупреждение",
                                   "Бубнов-Галеркин недоступен для пользовательских данных.")
            self.bubnov_galerkin_var.set(False)
            add_bubnov = False

        if not use_mkr_calc and not use_dqm_calc and not compare_methods_flag:
            messagebox.showerror("Ошибка",
                                 "Выберите хотя бы один метод (МКР/МДК) или сравнение.")
            return

        effective_use_mkr = use_mkr_calc or compare_methods_flag
        effective_use_dqm = use_dqm_calc or compare_methods_flag

        if compare_methods_flag and not (effective_use_mkr and effective_use_dqm):
            messagebox.showwarning("Предупреждение",
                                   "Для сравнения требуются результаты обоих методов (МКР и МДК).")

        self._clear_plot_and_results()
        self.calculation_results = None
        self.is_comparison_plot = False  # Сбрасываем флаг

        mkr_results_data = None
        dqm_results_data = None
        active_method_results = None
        method_name_for_display = ""

        # --- Получение параметров сетки ---
        try:
            mkr_grid_choice = self.mkr_grid_choice_var.get()
            if mkr_grid_choice == "Другая":
                grid_nx_mkr = int(self.mkr_custom_nx_var.get())
                grid_ny_mkr = int(self.mkr_custom_ny_var.get())
            else:
                grid_nx_mkr, grid_ny_mkr = self.PRESET_GRIDS[mkr_grid_choice]

            dqm_grid_choice = self.dqm_grid_choice_var.get()
            if dqm_grid_choice == "Другая":
                nx_intervals_dqm = int(self.dqm_custom_nx_var.get())
                ny_intervals_dqm = int(self.dqm_custom_ny_var.get())
            else:
                nx_intervals_dqm, ny_intervals_dqm = self.PRESET_GRIDS[dqm_grid_choice]

            if grid_nx_mkr <= 0 or grid_ny_mkr <= 0 or nx_intervals_dqm <= 0 or ny_intervals_dqm <= 0:
                messagebox.showerror("Ошибка сетки",
                                     "Размеры сетки должны быть положительными.")
                return

        except ValueError:
            messagebox.showerror("Ошибка сетки",
                                 "Введите корректные числовые значения для размеров сетки.")
            return

        # --- Расчеты ---
        if effective_use_mkr:
            print(f"Запуск МКР расчета (сетка {grid_nx_mkr}x{grid_ny_mkr})...")
            mkr_results_data = mkr_solver.solve_by_mkr(current_plate_params, boundary_condition,
                                                       grid_nx_mkr, grid_ny_mkr)
            if mkr_results_data:
                print("МКР результаты получены.")
                if not compare_methods_flag:
                    active_method_results = mkr_results_data
                    method_name_for_display = "МКР"
            elif not compare_methods_flag:
                messagebox.showerror("Ошибка МКР", "Расчет МКР не вернул результаты.")
                self._clear_plot_and_results()
                return
            elif compare_methods_flag:
                print("Предупреждение: МКР не вернул результаты для сравнения.")

        if effective_use_dqm:
            print(
                f"Запуск МДК расчета (интервалы {nx_intervals_dqm}x{ny_intervals_dqm}, тип: {self.dqm_grid_type_var.get()})...")
            dqm_solver_params = {'Nx_intervals': nx_intervals_dqm,
                                 'Ny_intervals': ny_intervals_dqm}
            dqm_results_data = dqm_solver.solve_by_dqm(current_plate_params, boundary_condition,
                                                       self.dqm_grid_type_var.get(),
                                                       dqm_solver_params)
            if dqm_results_data:
                print("МДК результаты получены.")
                if (not mkr_results_data or not effective_use_mkr) and not compare_methods_flag:
                    active_method_results = dqm_results_data
                    method_name_for_display = "МДК"
                messagebox.showerror("Ошибка МДК", "Расчет МДК не вернул результаты.")
                if not active_method_results: self._clear_plot_and_results(); return
            elif compare_methods_flag:
                print("Предупреждение: МДК не вернул результаты для сравнения.")

        # --- Отображение или Сравнение ---
        if compare_methods_flag:
            print("Запуск сравнения методов...")
            self.is_comparison_plot = True

            comparison_results = comparison_solver.run_convergence_study(
                plate_params=current_plate_params,
                boundary_condition=boundary_condition,
                dqm_grid_type_for_comparison=self.dqm_grid_type_var.get(),
                add_bubnov_data=add_bubnov
            )

            if comparison_results and comparison_results.get('plot_data'):
                self.calculation_results = comparison_results.get('plot_data')
                temp_plot_keys = []
                for base_key in CONVERGENCE_PLOT_KEYS_ORDER:
                    if f'{base_key}_mkr' in self.calculation_results or \
                            f'{base_key}_dqm' in self.calculation_results:
                        temp_plot_keys.append(base_key)
                self.current_plot_data_keys = temp_plot_keys

                if not self.current_plot_data_keys:
                    messagebox.showinfo("Сравнение",
                                        "Нет данных для отображения графиков сравнения.")
                else:
                    self.current_plot_index = 0
                    self._display_current_plot()
                    self.results_text_area.config(state="normal")
                    self.results_text_area.delete("1.0", tk.END)
                    self.results_text_area.insert(tk.END, comparison_results.get('summary_text',
                                                                                 "Сравнение завершено."))
                    self.results_text_area.config(state="disabled")
            else:
                messagebox.showerror("Ошибка сравнения",
                                     "Не удалось получить результаты для сравнения методов.")

        elif active_method_results:
            self.is_comparison_plot = False
            self.calculation_results = active_method_results
            self.current_plot_data_keys = [key for key in DEFAULT_PLOT_KEYS_ORDER if
                                           key in active_method_results and
                                           active_method_results[key] is not None]
            if not self.current_plot_data_keys:
                messagebox.showinfo("Нет данных",
                                    f"Метод {method_name_for_display} не вернул данных для отображения графиков.")
            else:
                self.current_plot_index = 0
                self._display_current_plot()
                self._update_numerical_results(active_method_results,
                                               method_name=method_name_for_display)
        else:
            self._clear_plot_and_results()

        self._update_plot_navigation_buttons_state()

        if save_to_word:
            self._export_plots_to_word()

    def _export_plots_to_word(self):
        if not self.calculation_results or not self.current_plot_data_keys:
            messagebox.showwarning("Экспорт в Word", "Нет данных для экспорта.")
            return

        plate_name_part = self.plate_type_var.get().replace(" ", "_")
        bc_part = self.boundary_condition_var.get()
        method_part = "MKR" if self.mkr_var.get() and not self.dqm_var.get() and not self.compare_methods_var.get() else \
            (
                "DQM" if self.dqm_var.get() and not self.mkr_var.get() and not self.compare_methods_var.get() else "Comparison")

        filename_suggestion = f"Результаты_{plate_name_part}_{bc_part}_{method_part}.docx"

        doc_dir = os.path.join("results", "documents")
        os.makedirs(doc_dir, exist_ok=True)

        filepath = filedialog.asksaveasfilename(
            initialdir=doc_dir,
            initialfile=filename_suggestion,
            defaultextension=".docx",
            filetypes=[("Word Documents", "*.docx"), ("All Files", "*.*")]
        )

        if not filepath:
            return

        try:
            figures_to_export = {}
            export_figsize = (7, 5.5)

            if self.is_comparison_plot:
                X_values = self.calculation_results.get('x_axis_data')
                if not X_values:
                    print(
                        "Предупреждение: нет 'x_axis_data' для графиков сходимости. Экспорт графиков сходимости невозможен.")

                for key_to_plot in self.current_plot_data_keys:
                    plot_info = PLOT_CONFIG.get(key_to_plot)
                    if not plot_info:
                        print(
                            f"Предупреждение: нет конфигурации для графика сходимости '{key_to_plot}'")
                        continue

                    if not X_values:
                        print(
                            f"Предупреждение: пропускаем график сходимости для '{key_to_plot}' из-за отсутствия x_axis_data.")
                        continue

                    temp_fig_conv = Figure(figsize=export_figsize, dpi=100)
                    temp_ax_conv = temp_fig_conv.add_subplot(111)

                    Y_mkr = self.calculation_results.get(f'{key_to_plot}_mkr')
                    Y_dqm = self.calculation_results.get(f'{key_to_plot}_dqm')
                    bubnov_val = self.calculation_results.get(f'{key_to_plot}_bubnov')

                    if Y_mkr is None and Y_dqm is None and bubnov_val is None:
                        print(
                            f"Предупреждение: нет Y данных (mkr, dqm или bubnov) для графика сходимости '{key_to_plot}'")

                        import matplotlib.pyplot as plt
                        plt.close(temp_fig_conv)
                        del temp_fig_conv
                        continue

                    plotter.plot_convergence_on_ax(
                        temp_ax_conv, X_values, Y_mkr, Y_dqm,
                        plot_info['title'], plot_info.get('xlabel', 'N'), plot_info['ylabel'],
                        "МКР", "МДК",
                        bubnov_value=bubnov_val if self.bubnov_galerkin_var.get() and self.plate_type_var.get() == "стандартная" else None,
                        multiplier=plot_info.get('multiplier', 1.0)
                    )
                    temp_fig_conv.tight_layout()
                    figures_to_export[plot_info['title']] = temp_fig_conv

            else:
                for key in self.current_plot_data_keys:
                    plot_info = PLOT_CONFIG.get(key)
                    if not plot_info:
                        print(f"Предупреждение: нет конфигурации для контурного графика '{key}'")
                        continue

                    X = self.calculation_results.get('X')
                    Y = self.calculation_results.get('Y')
                    Z = self.calculation_results.get(key)
                    if Z is None or X is None or Y is None:
                        print(
                            f"Предупреждение: нет данных (X, Y или Z) для контурного графика '{key}'")
                        continue

                    temp_fig_contour = Figure(figsize=export_figsize, dpi=100)
                    temp_ax_contour = temp_fig_contour.add_subplot(111)
                    Z_scaled = Z * plot_info.get('multiplier', 1.0)

                    plotter.plot_contour_on_ax(temp_ax_contour, X, Y, Z_scaled,
                                               plot_info['title'], plot_info['cbar'], key,
                                               cmap=plot_info.get('cmap', 'viridis'),
                                               levels=plot_info.get('levels', 50))
                    temp_fig_contour.tight_layout()
                    figures_to_export[plot_info['title']] = temp_fig_contour

            if not figures_to_export:
                messagebox.showinfo("Экспорт", "Нет графиков для экспорта.")
                return

            text_results = self.results_text_area.get("1.0", tk.END)

            word_exporter.export_to_word(filepath, figures_to_export, text_results,
                                         self.figure)
            messagebox.showinfo("Экспорт в Word", f"Результаты успешно сохранены в {filepath}")

        except Exception as e:
            messagebox.showerror("Ошибка экспорта", f"Не удалось сохранить файл: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import matplotlib.pyplot as plt
            for fig_title in figures_to_export:
                try:
                    plt.close(figures_to_export[fig_title])
                except:
                    pass


if __name__ == '__main__':
    app = Application()
    app.mainloop()
