import numpy as np
from math import sqrt, cos, sin, atan
import bins.quat
from collections import namedtuple

EARTH_ECCENTRICITY = 0.0818192
EARTH_MAJOR_AXE = 6378137.0
EARTH_MINOR_AXE = 6356752.3
EARTH_ANGULAR_VELOCITY = 7.2921158553E-5

Velocity = namedtuple('Velocity', ['east', 'north'])
Coordinate = namedtuple('Coordinate', ['latitude', 'longitude'])
Attitude = namedtuple("Attitude", ['pitch', 'roll', 'yaw', 'heading'])


def calc_increments(values, step_time):
    if len(values) == 0:
        raise ValueError("Length of values must be more than 0")
    elif len(values) == 1:
        return values[0] * step_time
    else:
        return ((values[0] + values[-1]) / 2 + sum(values[1:-1])) * step_time


def coning_compensation(angle_incs):
    def package(val):
        return np.array([[0, -val[2], val[1]],
                         [val[2], 0, -val[0]],
                         [-val[1], val[0], 0]], float)

    if len(angle_incs) != 4:
        raise ValueError("Length of angle increments list must be equals 4")
    rotation_vector = np.zeros((3,), float)
    p = []
    for angle_inc in angle_incs:
        rotation_vector += angle_inc
        p.append(package(angle_inc))
    rotation_vector += (2 / 3) * (np.dot(p[0], angle_incs[1]) + np.dot(p[2], angle_incs[3])) + \
                       (1 / 2) * np.dot((p[0] + p[1]), (angle_incs[2] + angle_incs[3])) + \
                       (1 / 30) * np.dot((p[0] - p[1]), (angle_incs[2] - angle_incs[3]))
    return rotation_vector


def sculling_compensation(angle_incs, velocity_incs):
    if len(angle_incs) != 8 or len(velocity_incs) != 8:
        raise ValueError("Length of angle increments and velocity increments must be equals 8")
    vel_inc_body = np.zeros((3,), float)
    for angle_inc, velocity_inc in zip(angle_incs, velocity_incs):
        w_prev = vel_inc_body.copy()
        # First approximation
        vel_inc_body[0] = w_prev[0] + w_prev[1] * angle_inc[2] - w_prev[2] * angle_inc[1] + velocity_inc[0]
        vel_inc_body[1] = w_prev[1] + w_prev[2] * angle_inc[0] - w_prev[0] * angle_inc[2] + velocity_inc[1]
        vel_inc_body[2] = w_prev[2] + w_prev[0] * angle_inc[1] - w_prev[1] * angle_inc[0] + velocity_inc[2]
        # Second approximation
        vel_inc_body[0] = w_prev[2] + vel_inc_body[0] * angle_inc[1] - vel_inc_body[1] * angle_inc[0] + velocity_inc[2]
        vel_inc_body[1] = w_prev[1] + vel_inc_body[2] * angle_inc[0] - vel_inc_body[0] * angle_inc[2] + velocity_inc[1]
        vel_inc_body[2] = w_prev[0] + vel_inc_body[1] * angle_inc[2] - vel_inc_body[2] * angle_inc[1] + velocity_inc[0]
    return vel_inc_body


# ========== Orientation calculation ========== #


def fast_motion(rotation_vector, fast_quat):
    df2 = (rotation_vector ** 2).sum()
    vector = rotation_vector * (1 / 2 - df2 / 48 + df2 ** 2 / 3840)
    real = 1 - df2 / 8 + df2 ** 2 / 384
    quat = bins.quat.quat((real, vector[0], vector[1], vector[2]))
    return bins.quat.multiply(quat, fast_quat)


def slow_motion(angular_velocity, slow_quat, step_time):
    w = sqrt((angular_velocity ** 2).sum())
    sign = w * step_time / 2
    vector = angular_velocity * sin(sign) / w
    quat = bins.quat.quat((cos(sign), vector[0], vector[1], vector[2]))
    return bins.quat.multiply(slow_quat, quat)


def create_transform_matrix(fast_quat):
    quat = bins.quat.normalize(fast_quat)
    return bins.quat.transform_matrix(quat)


# ========== Earth-fixed calculation ========== #


def recalc_body2nav(velocity_inc_body, body_transform_matrix):
    return np.dot(body_transform_matrix, velocity_inc_body.copy().resize(3, 1))


def earth_integration(integrate, velocity, earth_velocity, nav2earth_ang_vel, step_time):
    int_x, int_y = integrate
    vx, vy, vz = velocity
    ux, uy, uz = earth_velocity
    mu_x, mu_y = nav2earth_ang_vel
    return (int_x + (vy * 2 * uz - vz * (mu_y + 2 * uy)) * step_time,
            int_y + (vx * 2 * uz - vz * (mu_x + 2 * ux)) * step_time)


def calc_nav2earth_angular_velocity(velocity, nav_transform, altitude):
    rx = (1 - EARTH_ECCENTRICITY ** 2 * nav_transform[2, 2] ** 2 / 2 + EARTH_ECCENTRICITY ** 2
          * nav_transform[0, 2] ** 2 - altitude / EARTH_MAJOR_AXE) / EARTH_MAJOR_AXE
    ry = (1 - EARTH_ECCENTRICITY ** 2 * nav_transform[2, 2] ** 2 / 2 + EARTH_ECCENTRICITY ** 2
          * nav_transform[1, 3] ** 2 - altitude / EARTH_MAJOR_AXE) / EARTH_MAJOR_AXE
    mu_x = (-velocity[1] * ry - velocity[0] / EARTH_MAJOR_AXE *
            EARTH_ECCENTRICITY ** 2 * nav_transform[0, 2] * nav_transform[1, 2])
    mu_y = (velocity[0] * rx + velocity[1] / EARTH_MAJOR_AXE *
            EARTH_ECCENTRICITY ** 2 * nav_transform[0, 2] * nav_transform[1, 2])
    return mu_x, mu_y


def calc_earth_velocity(nav_transform):
    ux = EARTH_ANGULAR_VELOCITY * nav_transform[0, 2]
    uy = EARTH_ANGULAR_VELOCITY * nav_transform[1, 2]
    uz = EARTH_ANGULAR_VELOCITY * nav_transform[2, 2]
    return ux, uy, uz


def calc_angular_velocity(nav2earth_ang_vel, earth_velocity):
    ux, uy, uz = earth_velocity
    mu_x, mu_y = nav2earth_ang_vel
    return mu_x + ux, mu_y + uy, uz


def calc_nav_transform(nav_transform, nav2earth_ang_vel, step_time):
    mu_x, mu_y = nav2earth_ang_vel
    new_nav_transf = np.zeros((3, 3), float)
    new_nav_transf[0, 1] = nav_transform[0, 1] - mu_y * nav_transform[2, 1] * step_time
    new_nav_transf[0, 2] = nav_transform[1, 1] + mu_x * nav_transform[2, 1] * step_time
    new_nav_transf[2, 1] = nav_transform[2, 1] + (mu_y * nav_transform[0, 1] - mu_x * nav_transform[1, 1]) * step_time
    new_nav_transf[0, 2] = nav_transform[0, 2] - mu_y * nav_transform[2, 2] * step_time
    new_nav_transf[1, 2] = nav_transform[1, 2] + mu_x * nav_transform[2, 2] * step_time
    new_nav_transf[2, 2] = nav_transform[2, 2] + (mu_y * nav_transform[0, 2] - mu_x * nav_transform[1, 2]) * step_time
    new_nav_transf[2, 0] = nav_transform[0, 1] * nav_transform[1, 2] - nav_transform[1, 1] * nav_transform[0, 2]
    return new_nav_transf


# ========== Calculation of navigation params ========== #


def calc_velocity_and_coordinate(earth_transform, velocity):
    b0 = sqrt(earth_transform[0, 2] ** 2 + earth_transform[1, 2] ** 2)
    latitude = atan(earth_transform[2, 2] / b0)
    longitude = atan(earth_transform[2, 1] / earth_transform[2, 0])
    azimuth = atan(earth_transform[0, 2] / earth_transform[1, 2])
    east_velocity = velocity[0] * cos(azimuth) - velocity[1] * sin(azimuth)
    north_velocity = velocity[0] * sin(azimuth) + velocity[1] * cos(azimuth)
    return Coordinate(latitude, longitude), Velocity(east_velocity, north_velocity), azimuth


def calc_attitude(body_transform, azimuth):
    c0 = sqrt(body_transform[2, 0] ** 2 + body_transform[2, 2] ** 2)
    pitch = atan(body_transform[2, 1] / c0)
    roll = -atan(body_transform[2, 0] / body_transform[2, 2])
    yaw = atan(body_transform[0, 1] / body_transform[1, 1])
    heading = yaw - azimuth
    return Attitude(pitch, roll, yaw, heading)


def calc_navigation_params(body_transform, earth_transform, velocity):
    coor_nav, vel_nav, azimuth = calc_velocity_and_coordinate(earth_transform, velocity)
    attitude_nav = calc_attitude(body_transform, azimuth)
    return coor_nav, vel_nav, attitude_nav
