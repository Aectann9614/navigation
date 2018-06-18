from bins.utils import split_tee, window_count
from bins.math import calc_increments, sculling, coning, orientation_calc, earth_fixed_calc, calc_navigation_params


def bins_simple(initial, params, altitude):
    def navigation(data):
        def bins_algorithm(rotation, vel_inc, orientation, vel_and_coor):
            nonlocal ang_vel
            body_transform = orientation(rotation, ang_vel)
            tmp, ang_vel = vel_and_coor(vel_inc)
            earth_transform, vel = tmp
            return body_transform, earth_transform, vel
        ang_vel = initial['angular_velocity']

        acceler, gyro = split_tee(data, 2)
        velocity_inc = window_count(acceler, count, lambda val: calc_increments(val, step_1))
        angle_inc = window_count(gyro, count, lambda val: calc_increments(val, step_1))
        rotation_vector = window_count(window_count(angle_inc, 2, lambda val: sum(val)), 4, lambda val: coning(val))
        body_velocity_inc = window_count(zip(angle_inc, velocity_inc), 8, lambda val: sculling(val[0], val[1]))
        nav_res = (bins_algorithm(rotation, vel_inc,
                                  orientation_calc(initial['quaternion'], step_3),
                                  earth_fixed_calc(initial['transform'], initial['velocity'], altitude, step_3))
                   for rotation, vel_inc in zip(rotation_vector, body_velocity_inc))
        return (calc_navigation_params(body_transform, earth_transform, vel)
                for body_transform, earth_transform, vel in nav_res)
    count = params['acc_count']
    step_1 = params['data_step'] * count
    step_3 = step_1 * 8
    return navigation
