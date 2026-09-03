import math

def quaternion_to_euler(x, y, z, w):
    """
    euler angle
    """
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    #Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def calculate_unit_quaternion(xtrain):
    """
    calculate Unit Quaternion
    """

    xtrain['norm_quat'] = (xtrain['orientation_X'] ** 2 + xtrain['orientation_Y'] ** 2 
                           + xtrain['orientation_Z'] ** 2 + xtrain['orientation_W'] ** 2)
    
    xtrain['mod_quat'] = (xtrain['norm_quat']) ** 0.5

    xtrain['norm_X'] = xtrain['orientation_X'] / xtrain['mod_quat']
    xtrain['norm_Y'] = xtrain['orientation_Y'] / xtrain['mod_quat']
    xtrain['norm_Z'] = xtrain['orientation_Z'] / xtrain['mod_quat']
    xtrain['norm_W'] = xtrain['orientation_W'] / xtrain['mod_quat']

    return xtrain


def cal_euler_angle(X_train):

    unit = calculate_unit_quaternion(X_train)
    
    x, y, z, w = unit['norm_X'].tolist(), unit['norm_Y'].tolist(), unit['norm_Z'].tolist(), unit['norm_W'].tolist()

    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)

    unit['euler_x'] = nx
    unit['euler_y'] = ny
    unit['euler_z'] = nz
    return unit
