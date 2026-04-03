# app/calculations/comparison_solver.py
import numpy as np
import time
from . import mkr_solver
from . import dqm_solver
from . import plate_data

# Значения по умолчанию для сеток
DEFAULT_CONVERGENCE_MKR_GRIDS_INTERVALS = [(8, 6), (16, 8),
                                           (30, 15), (36, 18), (40, 20)]
# Для DQM передаем Nx_intervals, Ny_intervals
DEFAULT_CONVERGENCE_DQM_GRIDS_INTERVALS = [(8, 6), (16, 8),
                                           (30, 15), (36, 18), (40, 20)]


def _extract_metrics_from_results(results_dict, method_name=""):
    """Вспомогательная функция для извлечения метрик из словаря результатов солвера."""
    metrics = {}
    if results_dict is None or results_dict.get('displacements') is None:

        keys_to_nan = ['def_center', 'mx_max', 'my_max', 'mxy_max', 'qx_max', 'qy_max',
                       'sig_x_max', 'sig_y_max',
                       'tau_xy_max']
        for k_nan in keys_to_nan: metrics[k_nan] = np.nan
        return metrics

    displ = results_dict['displacements']  # (Ny, Nx)
    center_y, center_x = displ.shape[0] // 2, displ.shape[1] // 2
    metrics['def_center'] = displ[center_y, center_x]

    # Для МКР некоторые результаты могут быть не по всей области
    # np.nanmax(np.abs(...)) для корректного поиска максимума модуля
    metrics['mx_max'] = np.nanmax(np.abs(results_dict.get('Mx', np.nan)))
    metrics['my_max'] = np.nanmax(np.abs(results_dict.get('My', np.nan)))
    metrics['mxy_max'] = np.nanmax(
        np.abs(results_dict.get('Mxy', np.nan)))  # Mxy может быть знакопеременным

    # Qx, Qy могут быть неточными на самых краях для МКР
    metrics['qx_max'] = np.nanmax(np.abs(results_dict.get('Qx', np.nan)))
    metrics['qy_max'] = np.nanmax(np.abs(results_dict.get('Qy', np.nan)))

    metrics['sig_x_max'] = np.nanmax(np.abs(results_dict.get('sigma_x', np.nan)))
    metrics['sig_y_max'] = np.nanmax(np.abs(results_dict.get('sigma_y', np.nan)))
    metrics['tau_xy_max'] = np.nanmax(np.abs(results_dict.get('tau_xy', np.nan)))

    # Расчет эквивалентного напряжения на всякий случай
    # sx = results_dict.get('sigma_x')
    # sy = results_dict.get('sigma_y')
    # txy = results_dict.get('tau_xy')
    # if sx is not None and sy is not None and txy is not None:
    #     sigma_eq = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
    #     metrics['sig_eq_max'] = np.nanmax(np.abs(sigma_eq))
    # else:
    #     metrics['sig_eq_max'] = np.nan

    return metrics


def run_convergence_study(plate_params, boundary_condition,
                          mkr_grid_list=None,
                          dqm_grid_list_intervals=None,
                          dqm_grid_type_for_comparison="чебышева",
                          add_bubnov_data=False):
    if mkr_grid_list is None: mkr_grid_list = DEFAULT_CONVERGENCE_MKR_GRIDS_INTERVALS
    if dqm_grid_list_intervals is None: dqm_grid_list_intervals = DEFAULT_CONVERGENCE_DQM_GRIDS_INTERVALS

    results_mkr_convergence = []
    results_dqm_convergence = []

    print(f"\n--- Запуск исследования сходимости для ГУ: {boundary_condition} ---")
    print(f"--- DQM будет использовать тип узлов: {dqm_grid_type_for_comparison} ---")

    # --- МКР ---
    print("\n--- Сходимость МКР ---")
    current_mkr_grids_labels = []
    for nx_int, ny_int in mkr_grid_list:
        grid_label = f"{nx_int}x{ny_int}"
        current_mkr_grids_labels.append(grid_label)
        print(f"МКР: Расчет для сетки интервалов {nx_int}x{ny_int}...")
        start_time = time.time()
        mkr_res = mkr_solver.solve_by_mkr(plate_params, boundary_condition, nx_int, ny_int)
        mkr_time = time.time() - start_time

        data_point = {'grid_label': grid_label, 'time': mkr_time,
                      'N_total_nodes': (nx_int + 1) * (ny_int + 1)}
        extracted_metrics = _extract_metrics_from_results(mkr_res, "МКР")
        data_point.update(extracted_metrics)
        results_mkr_convergence.append(data_point)
        print(
            f"  Время: {mkr_time:.3f}с, Прогиб в центре: {data_point.get('def_center', np.nan) * 1000:.4f} мм")

    # --- МДК ---
    print("\n--- Сходимость МДК ---")
    current_dqm_grids_labels = []  # Метки для МДК могут отличаться, если dqm_grid_list_intervals другой
    for nx_int_dqm, ny_int_dqm in dqm_grid_list_intervals:
        # Метка для DQM по количеству УЗЛОВ
        grid_label_dqm = f"МДК({dqm_grid_type_for_comparison[0].upper()}) {(nx_int_dqm + 1)}x{(ny_int_dqm + 1)}"
        current_dqm_grids_labels.append(grid_label_dqm)

        dqm_specific_params = {'Nx_intervals': nx_int_dqm, 'Ny_intervals': ny_int_dqm}
        print(
            f"МДК: Расчет для сетки ({dqm_grid_type_for_comparison}) {nx_int_dqm + 1}x{ny_int_dqm + 1} узлов...")
        start_time = time.time()
        dqm_res = dqm_solver.solve_by_dqm(plate_params, boundary_condition,
                                          dqm_grid_type_for_comparison, dqm_specific_params)
        dqm_time = time.time() - start_time

        data_point_dqm = {'grid_label': grid_label_dqm, 'time': dqm_time,
                          'N_total_nodes': (nx_int_dqm + 1) * (ny_int_dqm + 1)}
        extracted_metrics_dqm = _extract_metrics_from_results(dqm_res, "МДК")
        data_point_dqm.update(extracted_metrics_dqm)
        results_dqm_convergence.append(data_point_dqm)
        print(
            f"  Время: {dqm_time:.3f}с, Прогиб в центре: {data_point_dqm.get('def_center', np.nan) * 1000:.4f} мм")

    # Формируем общую ось X для графиков
    plot_data = {
        'x_axis_data': current_mkr_grids_labels,  # Или более общая ось X
    }

    # Ключи метрик
    metric_source_keys = ['def_center', 'mx_max', 'my_max', 'mxy_max', 'qx_max', 'qy_max',
                          'sig_x_max', 'sig_y_max', 'tau_xy_max', 'time']

    for src_key in metric_source_keys:
        conv_key = f"{src_key}_conv"

        plot_data[f'{conv_key}_mkr'] = np.array(
            [r.get(src_key, np.nan) for r in results_mkr_convergence])
        plot_data[f'{conv_key}_dqm'] = np.array(
            [r.get(src_key, np.nan) for r in results_dqm_convergence])

    if add_bubnov_data and plate_params.get('name', '').startswith("Стандартная"):
        bubnov_data_dict = plate_data.get_bubnov_galerkin_data(boundary_condition)
        if bubnov_data_dict:
            bubnov_mapping = {

                'def_center': 'max_displacement', 'mx_max': 'mx_center',
                'my_max': 'my_center', 'mxy_max': 'mxy_center',
                'qx_max': 'Qx_max', 'qy_max': 'Qy_max',
                'sig_x_max': 'sigma_x_center', 'sig_y_max': 'sigma_y_center',
                'tau_xy_max': 'tau_xy_center',
            }
            for src_key, bubnov_data_key in bubnov_mapping.items():
                conv_key = f"{src_key}_conv"
                if bubnov_data_dict.get(bubnov_data_key) is not None:
                    plot_data[f'{conv_key}_bubnov'] = bubnov_data_dict[bubnov_data_key]

    summary = "Исследование сходимости завершено.\n"

    return {'plot_data': plot_data, 'summary_text': summary}
