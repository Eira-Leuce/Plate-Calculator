# app/calculations/plate_data.py
STANDARD_PLATE_PARAMS = {
    'length': 6.0,  # a м
    'width': 3.0,  # b м
    'thickness': 0.12,  # h м
    'young_modulus': 34.5e9,  # Па (сталь)
    'poisson_ratio': 0.2,  # (сталь)
    'load_intensity': 8000,  # Па (равномерно распределенная нагрузка)
    'name': "Стандартная стальная пластина 6x3м"
}

# Значения Бубнова-Галеркина для стандартной пластины
BUBNOV_GALERKIN_STANDARD_HINGED = {  # Для шарнирного опирания
    'max_displacement': 1.2696e-03,  # max w
    'mx_center': 2.8115e+03,  # center
    'my_center': 7.2325e+03,  # center
    'mxy_max': 3.6971e+03,  # abs max
    'mxy_center': 3.7971e+03,  # abs max
    'Qx_max': 7.3054e+03,  # abs max
    'Qy_max': 1.0532e+04,  # abs max
    'sigma_x_center': 1.1715e+06,  # abs max
    'sigma_y_center': 3.0135e+06,  # abs max
    'tau_xy_center': 1.5904e+06,  # abs max
}

BUBNOV_GALERKIN_STANDARD_CLAMPED = {  # Для жесткой заделки
    'max_displacement': 3.1743e-04,
    'mx_center': 2.9238e+03,
    'my_center': 5.2951e+03,
    'mxy_center': 9.0575e+02,
    'Qx_max': 7.7547e+03,
    'Qy_max': 1.2483e+04,
    'sigma_x_center': 1.2182e+06,
    'sigma_y_center': 2.2063e+06,
    'tau_xy_center': 3.7739e+05,
}


def get_bubnov_galerkin_data(boundary_condition):
    if boundary_condition == "шарнир":
        return BUBNOV_GALERKIN_STANDARD_HINGED
    elif boundary_condition == "заделка":
        return BUBNOV_GALERKIN_STANDARD_CLAMPED
    return None
