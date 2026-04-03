# app/calculations/dqm_solver.py
import numpy as np
from scipy.linalg import solve


# Вспомогательные функции


def chebyshev_gauss_lobatto_nodes(L, N_nodes):
    """Функция генерирует узлы Чебышева-Гаусса-Лобатто на интервале [0, L]"""
    if N_nodes == 1:
        return np.array([L / 2.0])
    i = np.arange(N_nodes)
    x_cheb = (L / 2.0) * (1.0 - np.cos(i * np.pi / (N_nodes - 1)))
    return x_cheb


def dqm_weights_physical(coords, order):
    """
        Расчет матриц весовых коэффициентов DQM для физических координат.
        coords: одномерный массив координат узлов.
        order: порядок производной (1, 2, ..., или 0 для единичной матрицы).
    """
    N_nodes = len(coords)
    if order == 0:
        return np.eye(N_nodes)

    # Проверка на достаточное количество узлов
    if N_nodes < 2 and order > 0:
        print(
            f"Ошибка DQM Weights: Для производной порядка {order} требуется как минимум {max(2, order)} узлов, получено {N_nodes}")
        return np.zeros((N_nodes, N_nodes))
    if N_nodes < order + 1 and order > 0:
        pass

    C_matrices = [np.zeros((N_nodes, N_nodes)) for _ in range(max(order + 1, 2))]

    C1 = np.zeros((N_nodes, N_nodes))
    for i in range(N_nodes):
        M_prime_xi = 1.0
        for k in range(N_nodes):
            if k != i:
                M_prime_xi *= (coords[i] - coords[k])
        sum_row_diag = 0.0
        for j in range(N_nodes):
            if i == j:
                continue
            M_prime_xj = 1.0
            for k_inner in range(N_nodes):
                if k_inner != j:
                    M_prime_xj *= (coords[j] - coords[k_inner])
            denominator = (coords[i] - coords[j]) * M_prime_xj
            if np.isclose(denominator, 0.0):
                C1[i, j] = 0.0
            else:
                C1[i, j] = M_prime_xi / denominator
            sum_row_diag += C1[i, j]
        C1[i, i] = -sum_row_diag
    C_matrices[1] = C1

    if order > 1:
        Ck_prev = C1.copy()
        for k_ord in range(2, order + 1):
            Ck_curr = np.dot(Ck_prev, C1)
            C_matrices[k_ord] = Ck_curr
            Ck_prev = Ck_curr
    return C_matrices[order]


# Функции для ОМДК (шарнирное опирание)
def get_global_idx(ix_node, iy_node, M_total_points_y):
    return ix_node * M_total_points_y + iy_node


def get_coords_from_global_idx(global_idx, M_total_points_y):
    ix_node = global_idx // M_total_points_y
    iy_node = global_idx % M_total_points_y
    return ix_node, iy_node


# Основная функция расчета МДК
def solve_by_dqm(plate_params, boundary_condition, grid_type, dqm_specific_params=None):
    try:
        # 1. Извлечение параметров и инициализация
        a = plate_params['length']
        b = plate_params['width']
        h_plate = plate_params['thickness']
        E = plate_params['young_modulus']
        mu_poisson = plate_params['poisson_ratio']
        q_load = plate_params['load_intensity']
        D_flex = E * h_plate ** 3 / (12 * (1 - mu_poisson ** 2))

        if dqm_specific_params is None:
            dqm_specific_params = {}
        Nx_intervals = dqm_specific_params.get('Nx_intervals', 30)
        Ny_intervals = dqm_specific_params.get('Ny_intervals', 20)
        N_points_x = Nx_intervals + 1
        M_points_y = Ny_intervals + 1
        total_nodes = N_points_x * M_points_y

        # 2. Генерация координат сетки
        if grid_type == "равномерная":
            x_coords = np.linspace(0, a, N_points_x)
            y_coords = np.linspace(0, b, M_points_y)
        elif grid_type == "чебышева":
            x_coords = chebyshev_gauss_lobatto_nodes(a, N_points_x)
            y_coords = chebyshev_gauss_lobatto_nodes(b, M_points_y)
        else:
            print(f"Ошибка DQM: Неизвестный тип сетки: {grid_type}")
            return None

        # 3. Генерация матриц весовых коэффициентов DQM
        Dx1 = dqm_weights_physical(x_coords, 1)
        Dx2 = dqm_weights_physical(x_coords, 2)
        Dx4 = dqm_weights_physical(x_coords, 4)
        Dy1 = dqm_weights_physical(y_coords, 1)
        Dy2 = dqm_weights_physical(y_coords, 2)
        Dy4 = dqm_weights_physical(y_coords, 4)

        # 4. Сборка глобальной матрицы оператора и правой части (до применения ГУ)
        Ix_identity = np.eye(N_points_x)
        Iy_identity = np.eye(M_points_y)
        L_op_full_matrix = np.kron(Dx4, Iy_identity) + \
                           2 * np.kron(Dx2, Dy2) + \
                           np.kron(Ix_identity, Dy4)
        f_rhs_full_vector = (q_load / D_flex) * np.ones(total_nodes)
        w_flat_solution = np.zeros(total_nodes)

        # 5. Применение граничных условий и решение
        if boundary_condition == "шарнир":
            if N_points_x < 5 or M_points_y < 5:
                print(f"Ошибка DQM (шарнир): Для ОМДК (классификация W_I/W_B) "
                      f"требуется Nx, Ny узлов >= 5. "
                      f"Получено: N_points_x={N_points_x}, M_points_y={M_points_y}. Увеличьте сетку.")
                return None

            map_global_to_type = {}
            map_global_to_local_idx = {}
            map_WI_to_global = []
            map_WB_to_global = []
            idx_I_count = 0
            idx_B_count = 0
            for ix in range(N_points_x):
                for iy in range(M_points_y):
                    glob_idx = get_global_idx(ix, iy, M_points_y)
                    if ix >= 2 and ix <= N_points_x - 3 and iy >= 2 and iy <= M_points_y - 3:
                        map_global_to_type[glob_idx] = 'I'
                        map_global_to_local_idx[glob_idx] = idx_I_count
                        map_WI_to_global.append(glob_idx)
                        idx_I_count += 1
                    else:
                        map_global_to_type[glob_idx] = 'B'
                        map_global_to_local_idx[glob_idx] = idx_B_count
                        map_WB_to_global.append(glob_idx)
                        idx_B_count += 1

            num_WI = len(map_WI_to_global)
            num_WB = len(map_WB_to_global)

            if num_WI == 0:
                print(f"Ошибка DQM (шарнир): После классификации нет W_I узлов (num_WI=0) "
                      f"при N_points_x={N_points_x}, M_points_y={M_points_y}. ОМДК неприменим.")
                return None

            D_II = np.zeros((num_WI, num_WI))
            D_IB = np.zeros((num_WI,
                             num_WB if num_WB > 0 else 1))
            F_I = np.zeros(num_WI)

            if num_WB > 0:
                D_BI = np.zeros((num_WB, num_WI))
                D_BB = np.zeros((num_WB, num_WB))
                F_B = np.zeros(num_WB)
            else:  # Случай отсутствия W_B узлов
                D_BI = np.empty((0, num_WI))
                D_BB = np.empty((0, 0))
                F_B = np.empty(0)

            for i_idx in range(num_WI):
                glob_idx_row = map_WI_to_global[i_idx]
                F_I[i_idx] = f_rhs_full_vector[glob_idx_row]
                for glob_idx_col in range(total_nodes):
                    val_L_op = L_op_full_matrix[glob_idx_row, glob_idx_col]
                    node_type_col = map_global_to_type.get(glob_idx_col)
                    local_idx_col = map_global_to_local_idx.get(glob_idx_col)
                    if node_type_col == 'I' and local_idx_col is not None:
                        D_II[i_idx, local_idx_col] = val_L_op
                    elif node_type_col == 'B' and local_idx_col is not None and num_WB > 0:
                        D_IB[i_idx, local_idx_col] = val_L_op

            if num_WB > 0:
                for b_idx in range(num_WB):
                    glob_idx_B_current_eq_node = map_WB_to_global[b_idx]
                    ix_b, iy_b = get_coords_from_global_idx(glob_idx_B_current_eq_node, M_points_y)
                    F_B[b_idx] = 0.0
                    if ix_b == 0 or ix_b == N_points_x - 1 or iy_b == 0 or iy_b == M_points_y - 1:
                        D_BB[b_idx, b_idx] = 1.0
                    elif ix_b == 1:
                        for kx in range(N_points_x):
                            val = Dx2[0, kx]
                            aff_idx = get_global_idx(kx, iy_b, M_points_y)
                            if map_global_to_type.get(aff_idx) == 'I':
                                D_BI[b_idx, map_global_to_local_idx[aff_idx]] += val
                            elif map_global_to_type.get(aff_idx) == 'B':
                                D_BB[b_idx, map_global_to_local_idx[aff_idx]] += val
                    elif ix_b == N_points_x - 2:
                        for kx in range(N_points_x):
                            val = Dx2[N_points_x - 1, kx]
                            aff_idx = get_global_idx(kx, iy_b, M_points_y)
                            if map_global_to_type.get(aff_idx) == 'I':
                                D_BI[b_idx, map_global_to_local_idx[aff_idx]] += val
                            elif map_global_to_type.get(aff_idx) == 'B':
                                D_BB[b_idx, map_global_to_local_idx[aff_idx]] += val
                    elif iy_b == 1:
                        for ky in range(M_points_y):
                            val = Dy2[0, ky]
                            aff_idx = get_global_idx(ix_b, ky, M_points_y)
                            if map_global_to_type.get(aff_idx) == 'I':
                                D_BI[b_idx, map_global_to_local_idx[aff_idx]] += val
                            elif map_global_to_type.get(aff_idx) == 'B':
                                D_BB[b_idx, map_global_to_local_idx[aff_idx]] += val
                    elif iy_b == M_points_y - 2:
                        for ky in range(M_points_y):
                            val = Dy2[M_points_y - 1, ky]
                            aff_idx = get_global_idx(ix_b, ky, M_points_y)
                            if map_global_to_type.get(aff_idx) == 'I':
                                D_BI[b_idx, map_global_to_local_idx[aff_idx]] += val
                            elif map_global_to_type.get(aff_idx) == 'B':
                                D_BB[b_idx, map_global_to_local_idx[aff_idx]] += val
            try:
                if num_WB > 0:
                    D_BB_inv_D_BI = solve(D_BB, D_BI, assume_a='gen')
                    D_BB_inv_F_B = np.zeros(
                        (num_WB, 1 if F_B.ndim == 1 or F_B.shape[0] == 0 else F_B.shape[1]))
                    if np.any(F_B): D_BB_inv_F_B = solve(D_BB, F_B[:,
                    np.newaxis] if F_B.ndim == 1 and F_B.size > 0 else F_B,
                                                         assume_a='gen')

                    K_eff_I = D_II - (
                        D_IB @ D_BB_inv_D_BI if D_IB.size > 0 and D_BB_inv_D_BI.size > 0 else np.zeros_like(
                            D_II))
                    F_eff_I_terms = F_I - ((
                                                   D_IB @ D_BB_inv_F_B).squeeze() if D_IB.size > 0 and D_BB_inv_F_B.size > 0 else np.zeros_like(
                        F_I))
                    W_I_solution = solve(K_eff_I, F_eff_I_terms, assume_a='gen')

                    RHS_for_WB = F_B - (
                        D_BI @ W_I_solution if D_BI.size > 0 and W_I_solution.size > 0 else np.zeros_like(
                            F_B))
                    W_B_solution = solve(D_BB, RHS_for_WB, assume_a='gen')

                    for i_idx_s in range(num_WI): w_flat_solution[map_WI_to_global[i_idx_s]] = \
                        W_I_solution[i_idx_s]
                    for b_idx_s in range(num_WB): w_flat_solution[map_WB_to_global[b_idx_s]] = \
                        W_B_solution[b_idx_s]
                else:  # Случай num_WI > 0, num_WB == 0 (теоретически, если все ГУ заданы на W_I)
                    W_I_solution = solve(D_II, F_I, assume_a='gen')  # L_II W_I = F_I
                    for i_idx_s in range(num_WI): w_flat_solution[map_WI_to_global[i_idx_s]] = \
                        W_I_solution[i_idx_s]


            except np.linalg.LinAlgError as e:
                print(f"Ошибка DQM (шарнир, ОМДК): Не удалось решить СЛАУ - {e}")
                return None

        elif boundary_condition == "заделка":
            if N_points_x < 3 or M_points_y < 3:
                print(f"Предупреждение DQM (заделка): Требуется N_points_x >= 3 и M_points_y >= 3. "
                      f"Получено N={N_points_x}, M={M_points_y}.")

            K_system_mod = L_op_full_matrix.copy()
            F_system_mod = f_rhs_full_vector.copy()
            is_row_modified_for_bc = np.zeros(total_nodes, dtype=bool)

            for ix in range(N_points_x):
                for iy in range(M_points_y):
                    if ix == 0 or ix == N_points_x - 1 or iy == 0 or iy == M_points_y - 1:
                        glob_idx = get_global_idx(ix, iy, M_points_y)
                        K_system_mod[glob_idx, :] = 0.0
                        K_system_mod[glob_idx, glob_idx] = 1.0
                        F_system_mod[glob_idx] = 0.0
                        is_row_modified_for_bc[glob_idx] = True

            for iy_g in range(M_points_y):
                if not (iy_g == 0 or iy_g == M_points_y - 1):
                    if N_points_x > 2:
                        idx_adj_L = get_global_idx(1, iy_g, M_points_y)
                        if not is_row_modified_for_bc[idx_adj_L]:
                            K_system_mod[idx_adj_L, :] = 0.0
                            for kx in range(N_points_x): K_system_mod[
                                idx_adj_L, get_global_idx(kx, iy_g, M_points_y)] = Dx1[0, kx]
                            F_system_mod[idx_adj_L] = 0.0
                            is_row_modified_for_bc[idx_adj_L] = True

                        idx_adj_R = get_global_idx(N_points_x - 2, iy_g, M_points_y)
                        if not is_row_modified_for_bc[idx_adj_R]:
                            K_system_mod[idx_adj_R, :] = 0.0
                            for kx in range(N_points_x): K_system_mod[
                                idx_adj_R, get_global_idx(kx, iy_g, M_points_y)] = Dx1[
                                N_points_x - 1, kx]
                            F_system_mod[idx_adj_R] = 0.0
                            is_row_modified_for_bc[idx_adj_R] = True

            for ix_g in range(N_points_x):
                if not (ix_g == 0 or ix_g == N_points_x - 1):
                    if M_points_y > 2:
                        idx_adj_B = get_global_idx(ix_g, 1, M_points_y)
                        if not is_row_modified_for_bc[idx_adj_B]:
                            K_system_mod[idx_adj_B, :] = 0.0
                            for ky in range(M_points_y): K_system_mod[
                                idx_adj_B, get_global_idx(ix_g, ky, M_points_y)] = Dy1[0, ky]
                            F_system_mod[idx_adj_B] = 0.0
                            is_row_modified_for_bc[idx_adj_B] = True

                        idx_adj_T = get_global_idx(ix_g, M_points_y - 2, M_points_y)
                        if not is_row_modified_for_bc[idx_adj_T]:
                            K_system_mod[idx_adj_T, :] = 0.0
                            for ky in range(M_points_y): K_system_mod[
                                idx_adj_T, get_global_idx(ix_g, ky, M_points_y)] = Dy1[
                                M_points_y - 1, ky]
                            F_system_mod[idx_adj_T] = 0.0
                            is_row_modified_for_bc[idx_adj_T] = True
            try:
                w_flat_solution = solve(K_system_mod, F_system_mod, assume_a='gen')
            except np.linalg.LinAlgError as e:
                print(f"Ошибка DQM (заделка): Не удалось решить СЛАУ - {e}")
                return None
        else:
            print(f"Ошибка DQM: Неизвестный тип граничного условия: {boundary_condition}")
            return None

        w_matrix = w_flat_solution.reshape((N_points_x, M_points_y))

        d2w_dx2_map = Dx2 @ w_matrix
        d2w_dy2_map = w_matrix @ Dy2.T
        d2w_dxdy_map = Dx1 @ (w_matrix @ Dy1.T)
        Mx_map = -D_flex * (d2w_dx2_map + mu_poisson * d2w_dy2_map)
        My_map = -D_flex * (d2w_dy2_map + mu_poisson * d2w_dx2_map)
        Mxy_map = -D_flex * (1 - mu_poisson) * d2w_dxdy_map
        lap_W_map = d2w_dx2_map + d2w_dy2_map
        d_lap_W_dx_map = Dx1 @ lap_W_map
        d_lap_W_dy_map = lap_W_map @ Dy1.T
        Qx_map = -D_flex * d_lap_W_dx_map
        Qy_map = -D_flex * d_lap_W_dy_map
        sigma_x_surf_map = 6 * Mx_map / h_plate ** 2
        sigma_y_surf_map = 6 * My_map / h_plate ** 2
        tau_xy_surf_map = 6 * Mxy_map / h_plate ** 2

        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')

        results = {
            'X': X_mesh.T, 'Y': Y_mesh.T, 'displacements': w_matrix.T,
            'Mx': Mx_map.T, 'My': My_map.T, 'Mxy': Mxy_map.T,
            'Qx': Qx_map.T, 'Qy': Qy_map.T,
            'sigma_x': sigma_x_surf_map.T,  # напряжения
            'sigma_y': sigma_y_surf_map.T,
            'tau_xy': tau_xy_surf_map.T,
            'x_coords': x_coords, 'y_coords': y_coords,
            'Nx_intervals': Nx_intervals,  # параметры сетки
            'Ny_intervals': Ny_intervals,
            'grid_type': grid_type,
            'D_flex': D_flex,
            'name': plate_params.get('name', "Пластина МДК")
        }
        return results

    except Exception as e:
        print(f"Критическая ошибка в dqm_solver.solve_by_dqm: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_plate_params = {
        'length': 6.0, 'width': 3.0, 'thickness': 0.12,
        'young_modulus': 34.5e9, 'poisson_ratio': 0.2,
        'load_intensity': 8.0e3, 'name': "Тестовая пластина DQM"
    }

    dqm_params_fine = {'Nx_intervals': 30, 'Ny_intervals': 20}
    dqm_params_coarse_ss = {'Nx_intervals': 4, 'Ny_intervals': 4}
    dqm_params_coarse_clamped = {'Nx_intervals': 2, 'Ny_intervals': 2}

    print("\n--- Тест DQM: Шарнир, равномерная, ГРУБАЯ (5x5 узлов) ---")
    res_ss_uni_c = solve_by_dqm(test_plate_params, "шарнир", "равномерная", dqm_params_coarse_ss)
    if res_ss_uni_c:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_ss_uni_c['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")

    print("\n--- Тест DQM: Заделка, равномерная, ГРУБАЯ (3x3 узла) ---")
    res_cl_uni_c = solve_by_dqm(test_plate_params, "заделка", "равномерная",
                                dqm_params_coarse_clamped)
    if res_cl_uni_c:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_cl_uni_c['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")

    print("\n--- Тест DQM: Шарнир, равномерная, ДЕТАЛЬНАЯ ---")
    res_ss_uni_f = solve_by_dqm(test_plate_params, "шарнир", "равномерная", dqm_params_fine)
    if res_ss_uni_f:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_ss_uni_f['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")

    print("\n--- Тест DQM: Шарнир, Чебышев, ДЕТАЛЬНАЯ ---")
    res_ss_cheb_f = solve_by_dqm(test_plate_params, "шарнир", "чебышева", dqm_params_fine)
    if res_ss_cheb_f:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_ss_cheb_f['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")

    print("\n--- Тест DQM: Заделка, равномерная, ДЕТАЛЬНАЯ ---")
    res_cl_uni_f = solve_by_dqm(test_plate_params, "заделка", "равномерная", dqm_params_fine)
    if res_cl_uni_f:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_cl_uni_f['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")

    print("\n--- Тест DQM: Заделка, Чебышев, ДЕТАЛЬНАЯ ---")
    res_cl_cheb_f = solve_by_dqm(test_plate_params, "заделка", "чебышева", dqm_params_fine)
    if res_cl_cheb_f:
        print(f"\tМакс. прогиб: {np.max(np.abs(res_cl_cheb_f['displacements'])):.4e} м")
    else:
        print("\tРасчет не удался.")
