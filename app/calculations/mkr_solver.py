# app/calculations/mkr_solver.py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_by_mkr(plate_params, boundary_condition, grid_nx, grid_ny):
    """
        Рассчитывает изгиб прямоугольной пластины методом конечных разностей (МКР).
    """
    try:
        # 1. Исходные данные и параметры сетки
        a = float(plate_params['length'])
        b = float(plate_params['width'])
        h_thick = float(plate_params['thickness'])
        E = float(plate_params['young_modulus'])
        mu_poisson = float(plate_params['poisson_ratio'])
        q_load = float(plate_params['load_intensity'])
        plate_name = plate_params.get('name', "Пластина МКР")  # Для вывода

        Nx = int(grid_nx)
        Ny = int(grid_ny)

        dx = a / Nx
        dy = b / Ny
        alpha_geom = dy / dx

        Nx_nodes = Nx + 1
        Ny_nodes = Ny + 1
        N_internal_nodes = (Nx - 1) * (Ny - 1)

        if N_internal_nodes <= 0:
            print("Ошибка МКР: Слишком маленькая сетка, нет внутренних узлов.")
            return None

        D_flex = E * h_thick ** 3 / (12 * (1 - mu_poisson ** 2))

        A = sp.lil_matrix((N_internal_nodes, N_internal_nodes))
        B = np.zeros(N_internal_nodes)
        load_term = (q_load * dx ** 4) / D_flex

        def get_k(i_node_idx, j_node_idx, Nx_intervals_local, Ny_intervals_local):
            return (i_node_idx - 1) * (Ny_intervals_local - 1) + (j_node_idx - 1)

        alpha2 = alpha_geom ** 2
        alpha4 = alpha_geom ** 4

        c00 = 6 + 8 / alpha2 + 6 / alpha4
        c10 = -4 - 4 / alpha2
        c01 = -4 / alpha2 - 4 / alpha4
        c11 = 2 / alpha2
        c20 = 1.0
        c02 = 1 / alpha4

        stencil_coeffs = {
            (0, 0): c00, (1, 0): c10, (-1, 0): c10, (0, 1): c01, (0, -1): c01,
            (1, 1): c11, (1, -1): c11, (-1, 1): c11, (-1, -1): c11,
            (2, 0): c20, (-2, 0): c20, (0, 2): c02, (0, -2): c02
        }

        for i_loop in range(1, Nx):
            for j_loop in range(1, Ny):
                k_eq_row = get_k(i_loop, j_loop, Nx, Ny)
                B[k_eq_row] = load_term

                for (di, dj), coeff_val in stencil_coeffs.items():
                    p_node, q_node = i_loop + di, j_loop + dj
                    target_p, target_q = p_node, q_node
                    sign_factor = 1.0
                    is_on_physical_boundary = False

                    if boundary_condition == "шарнир":
                        if p_node == 0 or p_node == Nx or q_node == 0 or q_node == Ny:
                            continue
                        elif p_node < 0:
                            target_p = 1
                            sign_factor = -1.0
                        elif p_node > Nx:
                            target_p = Nx - 1
                            sign_factor = -1.0
                        elif q_node < 0:
                            target_q = 1
                            sign_factor = -1.0
                        elif q_node > Ny:
                            target_q = Ny - 1
                            sign_factor = -1.0
                    elif boundary_condition == "заделка":
                        if p_node == 0 or p_node == Nx or q_node == 0 or q_node == Ny:
                            is_on_physical_boundary = True
                            continue
                        elif p_node < 0:
                            target_p = 1
                            sign_factor = 1.0
                        elif p_node > Nx:
                            target_p = Nx - 1
                            sign_factor = 1.0
                        elif q_node < 0:
                            target_q = 1
                            sign_factor = 1.0
                        elif q_node > Ny:
                            target_q = Ny - 1
                            sign_factor = 1.0
                    else:
                        raise ValueError(
                            f"Неизвестный тип граничного условия: {boundary_condition}")

                    if not is_on_physical_boundary:
                        if 0 < target_p < Nx and 0 < target_q < Ny:
                            l_unknown_col = get_k(target_p, target_q, Nx, Ny)
                            A[k_eq_row, l_unknown_col] += sign_factor * coeff_val

        A_csr = A.tocsr()
        if A_csr.shape[0] == 0:
            print("Ошибка МКР: Матрица A пуста.")
            return None

        W_internal_solution = spla.spsolve(A_csr, B)

        # Индексация w_full[y_node_idx, x_node_idx]
        w_matrix_full = np.zeros((Ny_nodes, Nx_nodes))  # (M, N)
        for i_idx_internal in range(1, Nx):  # x-координата внутреннего узла (1 до Nx-1)
            for j_idx_internal in range(1, Ny):  # y-координата внутреннего узла (1 до Ny-1)
                k_solution_idx = get_k(i_idx_internal, j_idx_internal, Nx, Ny)
                w_matrix_full[j_idx_internal, i_idx_internal] = W_internal_solution[k_solution_idx]

        # 2. Расчет производных и других величин
        if boundary_condition == "шарнир":
            def get_w_val(j_idx, i_idx, w_field):
                j_actual, i_actual, sign = j_idx, i_idx, 1.0
                if j_actual < 0:
                    j_actual = 1
                    sign *= -1.0
                elif j_actual >= Ny_nodes:
                    j_actual = Ny_nodes - 2
                    sign *= -1.0
                if i_actual < 0:
                    i_actual = 1
                    sign *= -1.0
                elif i_actual >= Nx_nodes:
                    i_actual = Nx_nodes - 2
                    sign *= -1.0
                return sign * w_field[j_actual, i_actual]
        elif boundary_condition == "заделка":
            def get_w_val(j_idx, i_idx, w_field):
                j_actual, i_actual = j_idx, i_idx
                if j_actual < 0:
                    j_actual = 1
                elif j_actual >= Ny_nodes:
                    j_actual = Ny_nodes - 2
                if i_actual < 0:
                    i_actual = 1
                elif i_actual >= Nx_nodes:
                    i_actual = Nx_nodes - 2
                return w_field[j_actual, i_actual]
        else:
            raise ValueError(f"Неизвестный ГУ: {boundary_condition}")

        d2w_dx2 = np.zeros_like(w_matrix_full)
        d2w_dy2 = np.zeros_like(w_matrix_full)
        d2w_dxdy = np.zeros_like(w_matrix_full)
        dx_sq, dy_sq, four_dx_dy = dx * dx, dy * dy, 4 * dx * dy

        for j in range(Ny_nodes):  # y-индекс
            for i in range(Nx_nodes):  # x-индекс
                w_ij = get_w_val(j, i, w_matrix_full)
                d2w_dx2[j, i] = (get_w_val(j, i + 1, w_matrix_full) - 2 * w_ij + get_w_val(j, i - 1,
                                                                                           w_matrix_full)) / dx_sq
                d2w_dy2[j, i] = (get_w_val(j + 1, i, w_matrix_full) - 2 * w_ij + get_w_val(j - 1, i,
                                                                                           w_matrix_full)) / dy_sq
                d2w_dxdy[j, i] = (get_w_val(j + 1, i + 1, w_matrix_full) - get_w_val(j + 1, i - 1,
                                                                                     w_matrix_full) -
                                  get_w_val(j - 1, i + 1, w_matrix_full) + get_w_val(j - 1, i - 1,
                                                                                     w_matrix_full)) / four_dx_dy

        Mx_vals = -D_flex * (d2w_dx2 + mu_poisson * d2w_dy2)
        My_vals = -D_flex * (d2w_dy2 + mu_poisson * d2w_dx2)
        Mxy_vals = -D_flex * (1 - mu_poisson) * d2w_dxdy

        lap_W = d2w_dx2 + d2w_dy2
        Qx_vals = np.zeros_like(w_matrix_full)
        Qy_vals = np.zeros_like(w_matrix_full)

        for j in range(Ny_nodes):
            Qx_vals[j, 0] = -D_flex * (get_w_val(j, 1, lap_W) - get_w_val(j, 0,
                                                                          lap_W)) / dx
            Qx_vals[j, Nx_nodes - 1] = -D_flex * (
                    get_w_val(j, Nx_nodes - 1, lap_W) - get_w_val(j, Nx_nodes - 2, lap_W)) / dx
            for i in range(1, Nx_nodes - 1):
                Qx_vals[j, i] = -D_flex * (
                        get_w_val(j, i + 1, lap_W) - get_w_val(j, i - 1, lap_W)) / (2 * dx)
        for i in range(Nx_nodes):
            Qy_vals[0, i] = -D_flex * (get_w_val(1, i, lap_W) - get_w_val(0, i, lap_W)) / dy
            Qy_vals[Ny_nodes - 1, i] = -D_flex * (
                    get_w_val(Ny_nodes - 1, i, lap_W) - get_w_val(Ny_nodes - 2, i, lap_W)) / dy
            for j in range(1, Ny_nodes - 1):
                Qy_vals[j, i] = -D_flex * (
                        get_w_val(j + 1, i, lap_W) - get_w_val(j - 1, i, lap_W)) / (2 * dy)

        sigma_x_vals = 6 * Mx_vals / h_thick ** 2
        sigma_y_vals = 6 * My_vals / h_thick ** 2
        tau_xy_vals = 6 * Mxy_vals / h_thick ** 2

        x_coords_1d = np.linspace(0, a, Nx_nodes)
        y_coords_1d = np.linspace(0, b, Ny_nodes)

        X_grid_final, Y_grid_final = np.meshgrid(x_coords_1d, y_coords_1d, indexing='xy')

        center_i_node = Nx // 2
        center_j_node = Ny // 2
        W_center_val = w_matrix_full[center_j_node, center_i_node]

        results = {
            'X': X_grid_final,
            'Y': Y_grid_final,
            'displacements': w_matrix_full,
            'Mx': Mx_vals,
            'My': My_vals,
            'Mxy': Mxy_vals,
            'Qx': Qx_vals,
            'Qy': Qy_vals,
            'sigma_x': sigma_x_vals,
            'sigma_y': sigma_y_vals,
            'tau_xy': tau_xy_vals,
            'W_max_val': W_center_val,
            'D_flex': D_flex,
            'Nx': Nx, 'Ny': Ny, 'dx': dx, 'dy': dy, 'name': plate_name
        }
        return results

    except Exception as e:
        print(f"Критическая ошибка в mkr_solver.solve_by_mkr: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("--- Тестирование модуля mkr_solver.py ---")
    test_plate_params_custom = {
        'length': 6.0, 'width': 3.0, 'thickness': 0.12,
        'young_modulus': 34.5e9, 'poisson_ratio': 0.2,
        'load_intensity': 8.0e3, 'name': "Тестовая МКР"
    }
    test_Nx = 30
    test_Ny = 20
    print("\n--- Тест 1: Шарнирное опирание (параметры из примера) ---")
    results_hinged = solve_by_mkr(test_plate_params_custom, "шарнир", test_Nx, test_Ny)
    if results_hinged:
        print(f"Прогиб в центре (шарнир): {results_hinged['W_max_val'] * 1000:.4f} мм")
        print(
            f"Форма X: {results_hinged['X'].shape}, Форма Y: {results_hinged['Y'].shape}, Форма w: {results_hinged['displacements'].shape}")
    print("\n--- Тест 2: Жесткое закрепление (параметры из примера) ---")
    results_clamped = solve_by_mkr(test_plate_params_custom, "заделка", test_Nx, test_Ny)
    if results_clamped:
        print(f"Прогиб в центре (заделка): {results_clamped['W_max_val'] * 1000:.4f} мм")
