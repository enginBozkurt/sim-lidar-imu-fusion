"""
Author: Miles Bennett
Generate LiDAR and IMU Simulation Data
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_imu_1d_data(T=10, fs=1e3, psd=1e-3):
    """
    Generate Data for 1D IMU
    
    Parameters
    ----------
    T: float
    time interval

    fs: float
    sampling frequency

    psd: float
    noise power spectral density [(m/s^2)/sqrt(Hz)]
    
    Returns: dict
    dictionary with the following keys
    time: vector of time values
    ground_truth: noiseless acceleration
    data: noisey acceleration data
    sigma_w: noise standard deviation
    """

    # calculate number of samples
    N = int(T*fs)+1

    # calculate time vector
    delta_t = 1/fs
    t_vec = delta_t*np.arange(0, N)

    # create acceleration profile
    x0 = 1
    td = t_vec[int(N/2)]
    tw = T/12
    a_vec = (x0/np.sqrt(tw**2*np.pi))*np.exp(-(t_vec-td)**2/(2*tw**2))*((t_vec-td)**2/tw**4-1/tw**2)

    # calculate noise standard deviation
    sigma_w = psd*np.sqrt(fs)

    # generate noise vector
    w_vec = sigma_w*np.random.randn(N)

    # generate noise acceleration data
    data = a_vec + w_vec

    # return results
    return {
        "time": t_vec,
        "ground_truth": a_vec,
        "data": data,
        "sigma_w": sigma_w
    }

def generate_data_lidar_1d(T=10, fs=10, sigma_w=0.02):
    """
    Generate 1D LiDAR Data
    Parameters
    ----------
    T: float
    time interval
    
    fs: float
    sampling time

    sigma_w: float
    standard deviation for LiDAR range estimate

    Returns: dict
    dictionary with the following keys
    time: time vector
    ground_truth: noiseless data
    data: time corrupted data
    sigma_w: noise standard deviation
    """

    # call imu generate data to ensure data is aligned
    L = 100 # upsampling factor
    imu_data = generate_imu_1d_data(T=T, fs=L*fs, psd=1e-3)

    # calculate ideal position
    t_vec = imu_data["time"]
    dt = t_vec[1]
    a_vec = imu_data["ground_truth"]
    p_vec = np.cumsum(np.cumsum(a_vec)*dt)*dt

    # downsample
    p_vec = p_vec[::L]
    t_vec = t_vec[::L]

    # create noise
    w_vec = sigma_w*np.random.randn(len(p_vec))

    # create noisy measurements
    data = p_vec + w_vec

    # return output
    return {
        "time": t_vec,
        "ground_truth": p_vec,
        "data": data,
        "sigma_w": sigma_w
    }

def test_1d_lidar():
    """
    Test 1d lidar
    """
    test_data = generate_data_lidar_1d(T=10, fs=10, sigma_w=0.02)
    plt.figure()
    plt.plot(test_data["time"], test_data["data"])
    plt.title("1D Lidar Test")

def generate_1d_data(
    outfile_imu="imu_data_1d.csv",
    outfile_lidar="lidar_data_1d.csv"):
    """
    Create 1D LiDAR Data Using Default Parameters
    """

    imu_data = generate_imu_1d_data()
    lidar_data = generate_data_lidar_1d()

    # create imu dataframe
    imu_df = pd.DataFrame()
    imu_df["time"] = imu_data["time"]
    imu_df["ax_measure"] = imu_data["data"]
    imu_df["ax_ideal"] = imu_data["ground_truth"]

    # create lidar dataframe
    lidar_df = pd.DataFrame()
    lidar_df["time"] = lidar_data["time"]
    lidar_df["px_measure"] = lidar_data["data"]
    lidar_df["px_ideal"] = lidar_data["ground_truth"]

    # save results
    lidar_df.to_csv(outfile_lidar, index=False)
    imu_df.to_csv(outfile_imu, index=False)

def generate_imu_2d_data(T=10, fs=1e3, psd=1e-3):
    """
    Generate Data for 2D IMU
    
    Parameters
    ----------
    T: float
    time interval

    fs: float
    sampling frequency

    psd: float
    noise power spectral density [(m/s^2)/sqrt(Hz)]
    
    Returns: dict
    dictionary with the following keys
    time: vector of time values
    ground_truth: noiseless acceleration
    ax: noisy acceleration data in x direction
    ay: noisy acceleration data in y direction
    sigma_w: noise standard deviation
    """

    # calculate number of samples
    N = int(T*fs)+1

    # calculate time vector
    delta_t = 1/fs
    t_vec = delta_t*np.arange(0, N)

    # create acceleration profile
    rx = 0.5
    ry = 0.5

    ax_ideal = -rx*(2*np.pi/T)**2*np.cos(2*np.pi*t_vec/T)
    ay_ideal = -ry*(2*np.pi/T)**2*np.sin(2*np.pi*t_vec/T)
    
    # calculate noise standard deviation
    sigma_w = psd*np.sqrt(fs)

    # generate noise vector
    wx = sigma_w*np.random.randn(N)
    wy = sigma_w*np.random.randn(N)

    # generate noise acceleration data
    ax = ax_ideal + wx
    ay = ay_ideal + wy

    # return results
    return {
        "time": t_vec,
        "ax_ideal": ax_ideal,
        "ay_ideal": ay_ideal,
        "ax_measure": ax,
        "ay_measure": ay,
        "sigma_w": sigma_w
    }

def calc_ideal_2d_pos(imu_data):
    """
    Calculate the ideal 2D position
    Assuming everything starts at rest

    imu_data: dictionary containing imu data
    """
    time = imu_data["time"]
    dt = time[1]
    ax = imu_data["ax_ideal"]
    ay = imu_data["ay_ideal"]
    
    # calculate velocity
    vx = np.cumsum(ax)*dt
    vy = np.cumsum(ay)*dt
    
    # adjust for initial conditions
    vy = vy - np.mean(vy)
    
    # calculate position
    px = np.cumsum(vx)*dt
    py = np.cumsum(vy)*dt
    
    # adjust for initial conditions
    px = px - np.mean(px)
    py = py - np.mean(py)
    
    # calculate initial state
    s0 = np.zeros(6)
    s0[0] = px[0]
    s0[1] = vx[0]
    s0[2] = ax[0]
    s0[3] = py[0]
    s0[4] = vy[0]
    s0[5] = ay[0]
    
    # return results
    return {
        "px": px,
        "py": py,
        "time": time,
        "s0": s0
    }

def generate_data_lidar_2d(T=10, fs=10, sigma_r=0.02, sigma_theta=0.001):
    """
    Generate 2D LiDAR Data
    Parameters
    ----------
    T: float
    time interval
    
    fs: float
    sampling time

    sigma_w: float
    standard deviation for LiDAR range estimate

    Returns: dict
    dictionary with the following keys
    time: time vector
    r_ideal: noiseless data (r)
    r: data (r)
    theta_ideal: noiseless data (theta)
    theta: data (theta)
    sigma_w: noise standard deviation
    """

    # generate ideal position
    L = 100 # upsampling factor
    imu_data = generate_imu_2d_data(T=10,fs=L*fs,psd=0)
    ideal_ins_pos = calc_ideal_2d_pos(imu_data)
    x = ideal_ins_pos["px"]
    y = ideal_ins_pos["py"]
    t = ideal_ins_pos["time"]
    # calculate range
    r_ideal = np.sqrt(x**2+y**2)
    theta_ideal = np.arctan2(y,x)

    # downsample
    r_ideal = r_ideal[::L]
    theta_ideal = theta_ideal[::L]
    t = t[::L]
    
    # create noise
    w_theta = sigma_theta*np.random.randn(len(t))
    w_r = sigma_r*np.random.randn(len(t))
    
    # create noisy measurements
    r = r_ideal + w_r
    theta = theta_ideal + w_theta

    # return output
    return {
        "time": t,
        "r_ideal": r_ideal,
        "theta_ideal": theta_ideal,
        "r_measure": r,
        "theta_measure": theta
    }

def test_2d_lidar():
    """
    Test 2D lidar with default parameters
    """
    lidar2d = generate_data_lidar_2d()
    plt.figure()
    x = lidar2d["r_measure"]*np.cos(lidar2d["theta_measure"])
    y = lidar2d["r_measure"]*np.sin(lidar2d["theta_measure"])
    plt.plot(x,y)
    plt.title("2D LiDAR Test")

def generate_2d_data(
    outfile_imu="imu_data_2d.csv",
    outfile_lidar="lidar_data_2d.csv",
    outfile_init_state = "init_state_2d.txt"):
    """
    Create 2D LiDAR Data Using Default Parameters
    """

    imu_data = generate_imu_2d_data()
    lidar_data = generate_data_lidar_2d()
    ideal_data = calc_ideal_2d_pos(imu_data)

    # create imu dataframe
    imu_df = pd.DataFrame()
    imu_df["time"] = imu_data["time"]
    imu_df["ax_measure"] = imu_data["ax_measure"]
    imu_df["ax_ideal"] = imu_data["ax_ideal"]
    imu_df["ay_measure"] = imu_data["ay_measure"]
    imu_df["ay_ideal"] = imu_data["ay_ideal"]

    # create lidar dataframe
    lidar_df = pd.DataFrame()
    lidar_df["time"] = lidar_data["time"]
    lidar_df["r_measure"] = lidar_data["r_measure"]
    lidar_df["r_ideal"] = lidar_data["r_ideal"]
    lidar_df["theta_measure"] = lidar_data["theta_measure"]
    lidar_df["r_ideal"] = lidar_data["r_ideal"]


    # create initial state
    s0 = ideal_data["s0"]

    # save results
    lidar_df.to_csv(outfile_lidar, index=False)
    imu_df.to_csv(outfile_imu, index=False)
    np.savetxt(outfile_init_state, s0)

if __name__ == "__main__":
    test_1d_lidar()
    generate_1d_data()
    generate_2d_data()
    test_2d_lidar()
    plt.show()