# app/postprocessing/plotter.py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contour_on_ax(ax, X, Y, Z_scaled, title, cbar_label, key_to_plot, levels=50,
                       cmap='viridis'):
    fig = ax.figure
    ax.clear()

    if hasattr(fig, "_colorbar_axes_list"):
        for cax_info in fig._colorbar_axes_list:
            try:
                ax_to_remove = cax_info['cax'] if isinstance(cax_info, dict) else cax_info
                if ax_to_remove in fig.axes: fig.delaxes(ax_to_remove)
            except (KeyError, AttributeError, ValueError):
                pass
        fig._colorbar_axes_list = []

    if X is None or Y is None or Z_scaled is None or X.size == 0 or Y.size == 0 or Z_scaled.size == 0:
        ax.text(0.5, 0.5, "Нет данных для отображения", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        fig.canvas.draw_idle()
        return

    contour = ax.contourf(X, Y, Z_scaled, levels=levels, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    if not hasattr(fig, "_colorbar_axes_list"): fig._colorbar_axes_list = []
    fig._colorbar_axes_list.append({'cax': cax})

    cb = fig.colorbar(contour, cax=cax, label=cbar_label)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x, м', fontsize=9)
    ax.set_ylabel('y, м', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

    if key_to_plot == 'displacements':
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_aspect('auto', adjustable='box')
    ax.grid(True, linestyle='-', color='gray', linewidth=0.5,
            alpha=0.7)
    try:
        fig.tight_layout(pad=1.2)
    except Exception:
        pass
    fig.canvas.draw_idle()


def plot_convergence_on_ax(ax, x_axis_data, y_data_mkr, y_data_dqm,
                           title, xlabel, ylabel,
                           label_mkr="МКР", label_dqm="МДК",
                           bubnov_value=None, multiplier=1.0,
                           annotate_points=False,
                           annotate_step=2):
    fig = ax.figure
    ax.clear()

    if x_axis_data is None or (y_data_mkr is None and y_data_dqm is None):
        ax.text(0.5, 0.5, "Нет данных для графика сходимости", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        fig.canvas.draw_idle()
        return

    plot_exists = False
    x_indices = np.arange(len(x_axis_data))

    all_y_values_for_scaling = []

    if y_data_mkr is not None and len(y_data_mkr) == len(x_axis_data):
        y_mkr_scaled = y_data_mkr * multiplier
        ax.plot(x_indices, y_mkr_scaled, 'o-', label=label_mkr)
        all_y_values_for_scaling.extend(
            y_mkr_scaled[np.isfinite(y_mkr_scaled)])
        if annotate_points:
            for i, txt_val in enumerate(y_mkr_scaled):
                if (i == 0 or (i % annotate_step == 0)) and np.isfinite(txt_val):
                    ax.annotate(f"МКР = {txt_val:.3e}", (x_indices[i], txt_val),
                                textcoords="offset points", xytext=(5, 5), ha='left', fontsize=7,
                                color='blue')
        plot_exists = True

    if y_data_dqm is not None and len(y_data_dqm) == len(x_axis_data):
        y_dqm_scaled = y_data_dqm * multiplier
        ax.plot(x_indices, y_dqm_scaled, 's--', label=label_dqm)
        all_y_values_for_scaling.extend(y_dqm_scaled[np.isfinite(y_dqm_scaled)])
        if annotate_points:
            for i, txt_val in enumerate(y_dqm_scaled):
                if (i == 0 or (i % annotate_step == 0)) and np.isfinite(txt_val):
                    ax.annotate(f"МДК = {txt_val:.3e}", (x_indices[i], txt_val),
                                textcoords="offset points", xytext=(5, -10), ha='left', fontsize=7,
                                color='green')
        plot_exists = True

    if bubnov_value is not None and not np.isnan(bubnov_value):
        bubnov_scaled = bubnov_value * multiplier
        label_bubnov = f'Бубнов-Галеркин ({bubnov_scaled:.3e})'
        ax.axhline(y=bubnov_scaled, color='black', linestyle='-', label=label_bubnov)
        if np.isfinite(bubnov_scaled):
            all_y_values_for_scaling.append(bubnov_scaled)
        plot_exists = True

    if not plot_exists:
        ax.text(0.5, 0.5, "Данные для построения отсутствуют или некорректны", ha='center',
                va='center', transform=ax.transAxes)
    else:
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(x) for x in x_axis_data], rotation=45, ha="right")
        ax.set_xlabel(xlabel if xlabel else "Сетка", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)

        # --- Установка пределов по оси Y ---
        if all_y_values_for_scaling:
            valid_y_values = np.array(all_y_values_for_scaling)
            valid_y_values = valid_y_values[np.isfinite(valid_y_values)]

            if valid_y_values.size > 0:
                y_min_data = np.min(valid_y_values)
                y_max_data = np.max(valid_y_values)

                reasonable_y_values = []
                if y_data_mkr is not None:
                    reasonable_y_values.extend(
                        (y_data_mkr * multiplier)[np.isfinite(y_data_mkr * multiplier)])
                if bubnov_value is not None and np.isfinite(bubnov_value * multiplier):
                    reasonable_y_values.append(bubnov_value * multiplier)

                if y_data_dqm is not None:
                    y_dqm_finite_scaled = (y_data_dqm * multiplier)[
                        np.isfinite(y_data_dqm * multiplier)]
                    if len(y_dqm_finite_scaled) > 0:
                        ref_max = -np.inf
                        ref_min = np.inf
                        if len(reasonable_y_values) > 0:
                            ref_max = np.max(reasonable_y_values) if len(
                                reasonable_y_values) > 0 else 0
                            ref_min = np.min(reasonable_y_values) if len(
                                reasonable_y_values) > 0 else 0

                        if not reasonable_y_values:
                            reasonable_y_values.extend(y_dqm_finite_scaled)
                        else:
                            for val_dqm in y_dqm_finite_scaled:
                                if val_dqm > ref_max * 100 and ref_max > 0: continue  # Сильный положительный выброс
                                if val_dqm < ref_min / 100 and ref_min < 0 and val_dqm < 0: continue  # Сильный отрицательный выброс
                                # Условие для выброса около нуля, если остальные значения не нулевые
                                if abs(val_dqm) < abs(ref_max / 1000) and abs(
                                        ref_max) > 1e-9 and abs(
                                    val_dqm) > 1e-15:  # DQM близко к нулю, а другие нет
                                    pass
                                elif abs(val_dqm) > abs(ref_max * 1000) and abs(
                                        ref_max) > 1e-15 and abs(
                                    val_dqm) > 1e-9:  # DQM очень большое, а другие нет
                                    continue
                                reasonable_y_values.append(val_dqm)

                if reasonable_y_values:
                    y_min_reasonable = np.min(reasonable_y_values)
                    y_max_reasonable = np.max(reasonable_y_values)
                    padding = (y_max_reasonable - y_min_reasonable) * 0.1
                    if padding == 0:  # Если все значения одинаковые
                        padding = abs(y_max_reasonable * 0.1) if y_max_reasonable != 0 else 0.1

                    final_y_min = y_min_reasonable - padding
                    final_y_max = y_max_reasonable + padding

                    ax.set_ylim(final_y_min, final_y_max)

                else:  # Если reasonable_y_values пуст
                    pass
            else:  # Если valid_y_values пуст
                pass
        else:  # Если all_y_values_for_scaling пуст
            pass

    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    try:
        fig.tight_layout(pad=1.2)
    except Exception:
        pass
    fig.canvas.draw_idle()
