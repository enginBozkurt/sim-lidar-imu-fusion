"""
Author: Miles Bennett
Algorithms for analyzing 1D and 2D LiDAR/IMU Data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Kalman_Filter_Scalar_Observation():
    """
    Class to Implement a Kalman Filter with a vector state, and 
    scalar observation
    """
    def __init__(self, **kwargs):
        self.A = kwargs.pop("A")
        self.B = kwargs.pop("B")
        self.H = kwargs.pop("H")
        self.Q = kwargs.pop("Q")
        self.C = kwargs.pop("C")

        self.initialize_filter()

    def initialize_filter(self):
        self.scc = np.zeros([len(self.H),1])
        self.Mcc = np.identity(len(self.scc))*10
        self.scp = 0*self.scc # Predicted State
        self.Mcp = 0*self.Mcc # Predicted MSE
        self.K = 0*self.scc #Kalman Gain
        self.I = np.identity(len(self.scc))
    def prediction(self):
        self.scp = np.dot(self.A, self.scc)
        self.Mcp = np.linalg.multi_dot([self.A, self.Mcc, self.A.T])+ \
            np.linalg.multi_dot([self.B,self.Q,self.B.T])
        K_num = np.dot(self.Mcp,self.H)
        K_den = self.C + np.linalg.multi_dot([self.H.T, self.Mcp, self.H])
        self.K = np.divide(K_num, K_den)

    def update(self, x):
        self.scc = self.scp + self.K*(x-np.dot(self.H.T,self.scp))
        Mcc_left = self.I - np.dot(self.K, self.H.T)
        self.Mcc = np.dot(Mcc_left, self.Mcp)

    def run(self, x):
        self.initialize_filter()
        filter_px = np.zeros(len(x))
        filter_vx = np.zeros(len(x))
        filter_ax = np.zeros(len(x))
        for ii in range(len(x)):
            x_cur = x[ii]
            self.prediction()
            self.update(x_cur)
            filter_px[ii] = self.scc[0]
            filter_vx[ii] = self.scc[1]
            filter_ax[ii] = self.scc[2]
        return {
            "px": filter_px,
            "vx": filter_vx,
            "ax": filter_ax
        }

class Kalman_1D_fusion():
    """
    Kalman Filter to fuse 1D lidar and IMU data
    Note IMU data and LiDAR dat are at different rates
    """
    def __init__(self, **kwargs):
        self.A = kwargs.pop("A")
        self.B = kwargs.pop("B")
        self.H = kwargs.pop("H")
        self.Q = kwargs.pop("Q")
        self.C = kwargs.pop("C")
        self.d_factor = kwargs.pop("d_factor")
        self.init_filter()
    def init_filter(self):
        self.scc = np.zeros([self.A.shape[0],1])
        self.Mcc = np.identity(self.A.shape[0])*10
        self.scp = 0*self.scc # Predicted State
        self.Mcp = 0*self.Mcc # Predicted MSE
        self.K = 0*self.scc #Kalman Gain
        self.I = np.identity(len(self.scc))
    def prediction(self):
        self.scp = np.dot(self.A, self.scc)
        self.Mcp = np.linalg.multi_dot([self.A, self.Mcc, self.A.T])+ \
            np.linalg.multi_dot([self.B,self.Q,self.B.T])
        K_left = np.dot(self.Mcp,self.H.T)
        K_right = np.linalg.inv(self.C + np.linalg.multi_dot([self.H, self.Mcp, self.H.T]))
        self.K = np.dot(K_left, K_right)
    def update(self, x):
        self.scc = self.scp + np.dot(self.K,(x-np.dot(self.H,self.scp)))
        self.Mcc = np.dot(self.I - np.dot(self.K, self.H),self.Mcp)
    def run(self, px, ax):
        N = len(ax)
        self.init_filter()
        filter_px = np.zeros(N)
        filter_vx = np.zeros(N)
        filter_ax = np.zeros(N)
        for ii in range(N):
            if ii%self.d_factor==0:
                self.H[0,0]=1
                jj = int(ii/self.d_factor)
                px_cur = px[jj]
            else:
                self.H[0,0]=0
                px_cur = 0 # set to 0 or any random value
            ax_cur = ax[ii]
            x_cur = np.array([[px_cur],[ax_cur]])
            self.prediction()
            self.update(x_cur)
            filter_px[ii] = self.scc[0]
            filter_vx[ii] = self.scc[1]
            filter_ax[ii] = self.scc[2]
        return {
            "px": filter_px,
            "vx": filter_vx,
            "ax": filter_ax,
        }

def interpolate_1d_lidar(
    lidar_datafile="lidar_data_1d.csv",
    imu_datafile="imu_data_1d.csv"):

    """
    Interpolate 1d LiDAR data at IMU rate
    """
    # load data
    imu_data = pd.read_csv(imu_datafile)
    imu_t_vec = imu_data["time"]

    lidar_data = pd.read_csv(lidar_datafile)
    lidar_t_vec = lidar_data["time"]
    px_data = lidar_data["px_measure"]

    px_itp = np.interp(imu_t_vec, lidar_t_vec, px_data)

    # return results
    return {
        "px": px_itp,
        "time": imu_t_vec
    }

def run_imu_kf(datafile="imu_data_1d.csv", sigma_w=0.1):
    """
    Run Kalman Filter on IMU data to estimate position
    
    Parameters
    ----------
    datafile: str 
    datafile for IMU data

    sigma_w: float
    standard deviation of the noise

    Returns: dictionary
    time: time vector
    px: position estimate
    vx: velocity estimate
    ax: acceleration estimate
    """

    # load data
    data = pd.read_csv(datafile)
    t_vec = data["time"]
    ax_data = data["ax_measure"]
    dt = t_vec[1]

    # define matrices for Kalman Filter
    A = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
    B = np.array([[0], [0], [1]])
    H = np.array([[0], [0], [1]])
    Q = np.array([[0.01]])
    C = sigma_w**2

    params = {
        "A": A,
        "B": B,
        "H": H,
        "Q": Q,
        "C": C
    }
    KF = Kalman_Filter_Scalar_Observation(**params)

    # filter data
    filter_data = KF.run(ax_data)
    filter_data["time"] = t_vec
    # return results
    return filter_data

def run_1d_KF_fusion(
    imu_datafile="imu_data_1d.csv", 
    imu_sigma_w=0.1,
    lidar_datafile="lidar_data_1d.csv",
    lidar_sigma_w=0.01):
    """
    Fuse 1D IMU and LiDAR data using a Kalman Filter
    """

    # load data
    imu_data = pd.read_csv(imu_datafile)
    imu_t_vec = imu_data["time"]
    ax_data = imu_data["ax_measure"]
    dt_imu = imu_t_vec[1]

    lidar_data = pd.read_csv(lidar_datafile)
    lidar_t_vec = lidar_data["time"]
    px_data = lidar_data["px_measure"]
    dt_lidar = lidar_t_vec[1]

    # calculate downsampling factor
    d_factor = int(dt_lidar/dt_imu)

    # define matrices of Kalman Filter
    A = np.array([[1, dt_imu, 0.5*dt_imu**2], [0, 1, dt_imu], [0, 0, 1]])
    B = np.array([[0], [0], [1]])
    H = np.array([[1, 0, 0], [0, 0, 1]])
    Q = np.array([[1e-4]])
    C = np.array([[lidar_sigma_w**2, 0], [0, imu_sigma_w**2]])
    C = np.array([[lidar_sigma_w**2, 0], [0, imu_sigma_w**2]])

    params = {
        "A": A,
        "B": B,
        "H": H,
        "Q": Q,
        "C": C,
        "d_factor": d_factor
    }
    KF_fuse = Kalman_1D_fusion(**params)

    # Run Kalman Filter
    filter_data = KF_fuse.run(px_data, ax_data)
    filter_data["time"] = imu_t_vec

    return filter_data


def calc_ideal_1d_pos(datafile="imu_data_1d.csv"):
    """
    Calculate ideal 1d position, velocity, acceleration
    """
    # load data
    data = pd.read_csv(datafile)
    t_vec = data["time"]
    ax_ideal = data["ax_ideal"]
    dt = t_vec[1]

    vx_ideal = np.cumsum(ax_ideal)*dt
    px_ideal = np.cumsum(vx_ideal)*dt

    # return results
    return {
        "time": t_vec,
        "ax": ax_ideal,
        "vx": vx_ideal,
        "px": px_ideal
    }

def test_imu_kf():
    """
    Test run_imu_kf with default parameters
    """
    test_data = run_imu_kf()
    ideal_data = calc_ideal_1d_pos()
    plt.figure()
    plt.plot(test_data["time"], test_data["px"], label="estimate")
    plt.plot(ideal_data["time"], ideal_data["px"], label="ideal")
    plt.legend()
    plt.title("Test IMU KF")

def test_1d_sensor_fusion_kf():
    """
    Test 1D Sensor Fusion KF with default parameters
    """
    test_data = run_1d_KF_fusion()
    ideal_data = calc_ideal_1d_pos()
    plt.figure()
    plt.plot(test_data["time"], test_data["px"], label="estimate")
    plt.plot(ideal_data["time"], ideal_data["px"], label="ideal")
    plt.legend()
    plt.title("Test IMU KF")

def compare_1d_results():
    """
    Compare Results for 1D problem
    """
    imu_int_data = run_imu_kf()
    lidar_itp_data = interpolate_1d_lidar()
    fusion_data = run_1d_KF_fusion()
    ideal_data = calc_ideal_1d_pos()
    
    plt.figure()
    plt.plot(ideal_data["time"], ideal_data["px"], label="Ideal",linewidth=2, color="black")
    plt.plot(imu_int_data["time"], imu_int_data["px"],"--",label="IMU only", linewidth=2, color="blue")
    plt.plot(lidar_itp_data["time"], lidar_itp_data["px"],"-.",label="LiDAR only", linewidth=2, color="orange")
    plt.plot(fusion_data["time"], fusion_data["px"], label="Fusion",linewidth=2, color="green")
    plt.legend()
    plt.title("Motion Trajectory: 1D Algorithms Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    plt.tight_layout()

    # calculate error
    itp_error = lidar_itp_data["px"] - ideal_data["px"]
    imu_error = imu_int_data["px"] - ideal_data["px"]
    fusion_error = fusion_data["px"] - ideal_data["px"]

    # print RMS distance error for each algorithm
    rms_itp = np.sqrt(sum(itp_error**2))
    rms_imu = np.sqrt(sum(imu_error**2))
    rms_fusion = np.sqrt(sum(fusion_error**2))

    print("RMS LIDAR: " + str(rms_itp))
    print("RMS IMU: " + str(rms_imu))
    print("RMS FUSION: " + str(rms_fusion))

    # plot results
    plt.figure()
    plt.plot(imu_int_data["time"], imu_error,"--",label="IMU only", linewidth=2, color="blue")
    plt.plot(lidar_itp_data["time"], itp_error,"-.",label="LiDAR only", linewidth=2, color="orange")
    plt.plot(fusion_data["time"], fusion_error, label="Fusion",linewidth=2, color="green")
    plt.legend()
    plt.title("Distance Error: 1D Algorithms Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.grid()
    plt.tight_layout()


class Kalman_Filter_Vector_Observation():
    """
    Class to Implement a Kalman Filter with a vector state, and 
    vector observation
    """
    def __init__(self, **kwargs):
        self.A = kwargs.pop("A")
        self.B = kwargs.pop("B")
        self.H = kwargs.pop("H")
        self.Q = kwargs.pop("Q")
        self.C = kwargs.pop("C")
        self.scc = kwargs.pop("s0") #initial state
        
        # reshape initial state
        self.scc = np.reshape(self.scc, (len(self.scc),1))
        self.init_filter()
    def init_filter(self):
        self.Mcc = np.identity(self.A.shape[0])*0.05
        self.scp = 0*self.scc # Predicted State
        self.Mcp = 0*self.Mcc # Predicted MSE
        self.K = 0*self.scc #Kalman Gain
        self.I = np.identity(len(self.scc))
    def prediction(self):
        self.scp = np.dot(self.A, self.scc)
        self.Mcp = np.linalg.multi_dot([self.A, self.Mcc, self.A.T])+ \
            np.linalg.multi_dot([self.B,self.Q,self.B.T])
        K_left = np.dot(self.Mcp,self.H.T)
        K_right = np.linalg.inv(self.C + np.linalg.multi_dot([self.H, self.Mcp, self.H.T]))
        self.K = np.dot(K_left, K_right)
    def update(self, x):
        self.scc = self.scp + np.dot(self.K,(x-np.dot(self.H,self.scp)))
        self.Mcc = np.dot(self.I - np.dot(self.K, self.H),self.Mcp)
    def run(self, x):
        N,M = x.shape
        self.init_filter()
        filter_out = np.zeros([N, len(self.scc)])
        for ii in range(N):
            x_cur = np.reshape(x[ii,:],(M,1))
            self.prediction()
            self.update(x_cur)
            filter_out[ii,:] = (self.scc).flatten()
        return {
            "px": filter_out[:,0],
            "vx": filter_out[:,1],
            "ax": filter_out[:,2],
            "py": filter_out[:,3],
            "vy": filter_out[:,4],
            "ay": filter_out[:,5]
        }

def calc_ideal_2d_pos(datafile="imu_data_2d.csv"):
    """
    Calculate the ideal 2D position
    Assuming everything starts at rest

    imu_data: dictionary containing imu data
    """
    imu_data = pd.read_csv(datafile)
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

def run_2d_imu_kf(
    datafile="imu_data_2d.csv",
    init_state_file = "init_state_2d.txt", 
    sigma_w=0.1):
    """
    Run and test 2D IMU Kalman Filter
    """

    # load data
    imu_data = pd.read_csv(datafile)
    t_vec = imu_data["time"]
    dt = t_vec[1]
    s0 = np.loadtxt(init_state_file)

    # define matrices
    A = np.array([[1, dt, 0.5*dt**2, 0, 0, 0], 
              [0, 1, dt, 0, 0, 0], 
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
    B = np.array([[0, 0], 
              [0, 0], 
              [1, 0],
              [0, 0],
              [0, 0],
              [0, 1]])
    H = np.array([[0, 0, 1, 0, 0, 0], 
              [0, 0, 0, 0, 0, 1]])
    Q = np.array([[0.01, 0],
              [0, 0.01]])
    C = np.array([[sigma_w**2, 0],
                [0, sigma_w**2]])
    params = {
    "A": A,
    "B": B,
    "H": H,
    "Q": Q,
    "C": C,
    "s0": s0
    }
    KF2D = Kalman_Filter_Vector_Observation(**params)

    # create data vector
    input_data = np.vstack([imu_data["ax_measure"], imu_data["ay_measure"]]).T

    # run filter
    filter_data = KF2D.run(input_data)
    filter_data["time"] = t_vec
    
    # return results
    return filter_data

def test_imu_2d_kf():
    """
    test default settings for 2D imu Kalman Filter
    """
    filter_data = run_2d_imu_kf()
    ideal_data = calc_ideal_2d_pos()

    # plot results
    plt.figure()
    plt.plot(filter_data["px"], filter_data["py"], label="IMU only")
    plt.plot(ideal_data["px"], ideal_data["py"], label="ideal")
    plt.legend()
    plt.title("IMU Only")

class Extended_Kalman_Filter_2D_Fusion():
    """
    Class to Implement a Kalman Filter with a vector state, and 
    vector observation. Fuses 2D LiDAR data and IMU data.
    """
    def __init__(self, **kwargs):
        self.A = kwargs.pop("A")
        self.B = kwargs.pop("B")
        self.Q = kwargs.pop("Q")
        self.C = kwargs.pop("C")
        self.scc = kwargs.pop("s0") #initial state
        # downsampling factor
        self.d_factor = kwargs.pop("d_factor")
        
        # indicate if LiDAR data is available
        self.lidar_ready = True
        # reshape initial state
        self.scc = np.reshape(self.scc, (len(self.scc),1))
        self.init_filter()
        
    def init_filter(self):
        self.Mcc = np.identity(self.A.shape[0])*0.05
        self.scp = 0*self.scc # Predicted State
        self.Mcp = 0*self.Mcc # Predicted MSE
        self.K = 0*self.scc #Kalman Gain
        self.I = np.identity(len(self.scc))
    
        # values are hard coded because the implementation
        # is specific to this problem
        self.H = np.zeros([4,6])
    # nonlinear observation function
    def h(self):
        hx = np.zeros(4)
        hx[0] = self.scp[2,0]
        hx[1] = self.scp[5,0]
        if self.lidar_ready:
            x = self.scp[0,0]
            y = self.scp[3,0]
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            hx[2] = r
            hx[3] = theta
        hx = np.reshape(hx, (len(hx),1))
        return hx
    # linearized observation Matrix update
    def calc_H(self):
        self.H = np.zeros([4,6])
        self.H[0,2] = 1
        self.H[1,5] = 1
        if self.lidar_ready:
            x = self.scp[0]
            y = self.scp[3]
            r = np.sqrt(x**2+y**2)
            self.H[2,0] = x/r
            self.H[2,3] = y/r
            self.H[3,0] = -y/r**2
            self.H[3,3] = x/r**2
    def prediction(self):
        self.scp = np.dot(self.A, self.scc)
        self.calc_H()
        self.Mcp = np.linalg.multi_dot([self.A, self.Mcc, self.A.T])+ \
            np.linalg.multi_dot([self.B,self.Q,self.B.T])
        K_left = np.dot(self.Mcp,self.H.T)
        K_right = np.linalg.inv(self.C + np.linalg.multi_dot([self.H, self.Mcp, self.H.T]))
        self.K = np.dot(K_left, K_right)
    def update(self, x):
        hx = self.h()
        self.scc = self.scp + np.dot(self.K,x-hx)
        self.Mcc = np.dot(self.I - np.dot(self.K, self.H),self.Mcp)
    def run(self, imu_data, lidar_data):
        N,M = imu_data.shape
        self.init_filter()
        filter_out = np.zeros([N, len(self.scc)])
        for ii in range(N):
            imu_cur = imu_data[ii,:]
            self.lidar_ready = (ii%self.d_factor==0)
            #self.lidar_ready = True
            if self.lidar_ready:
                jj = int(ii/self.d_factor)
                lidar_cur = lidar_data[jj,:]
            else:
                # default value
                lidar_cur = lidar_data[0,:]
            x_cur = np.hstack([imu_cur, lidar_cur])
            x_cur = np.reshape(x_cur, (len(x_cur),1))
            self.prediction()
            self.update(x_cur)
            filter_out[ii,:] = (self.scc).flatten()
        return {
            "px": filter_out[:,0],
            "vx": filter_out[:,1],
            "ax": filter_out[:,2],
            "py": filter_out[:,3],
            "vy": filter_out[:,4],
            "ay": filter_out[:,5]
        }

def run_2d_imu_lidar_ekf(
    imu_datafile="imu_data_2d.csv",
    lidar_datafile="lidar_data_2d.csv",
    init_state_file = "init_state_2d.txt", 
    sigma_ax=0.1,
    sigma_ay=0.1,
    sigma_r=0.06,
    sigma_theta=0.01):
    """
    Run and test 2D IMU Kalman Filter
    """

    # load data
    imu_data = pd.read_csv(imu_datafile)
    lidar_data = pd.read_csv(lidar_datafile)
    lidar_t_vec = lidar_data["time"]
    lidar_dt = lidar_t_vec[1]
    lidar_input = np.vstack([lidar_data["r_measure"], lidar_data["theta_measure"]]).T
    
    imu_t_vec = imu_data["time"]
    imu_dt = imu_t_vec[1]
    dt = imu_dt
    imu_input = np.vstack([imu_data["ax_measure"], imu_data["ay_measure"]]).T

    # calculate downsampling factor
    d_factor = int(lidar_dt/imu_dt)

    # load initial conditions
    s0 = np.loadtxt(init_state_file)

    # define matrices
    A = np.array([[1, dt, 0.5*dt**2, 0, 0, 0], 
              [0, 1, dt, 0, 0, 0], 
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
    B = np.array([[0, 0], 
              [0, 0], 
              [1, 0],
              [0, 0],
              [0, 0],
              [0, 1]])
    Q = np.array([[0.01, 0],
              [0, 0.01]])
    C = np.array([[sigma_ax**2, 0, 0, 0], 
              [0, sigma_ay**2, 0, 0],
              [0, 0, sigma_r**2, 0],
              [0, 0, 0, sigma_theta**2]])
    params = {
        "A": A,
        "B": B,
        "Q": Q,
        "C": C,
        "s0": s0,
        "d_factor": d_factor
    }
    EKF2D = Extended_Kalman_Filter_2D_Fusion(**params)

    ekf_out = EKF2D.run(imu_input, lidar_input)
    ekf_out["time"] = imu_t_vec
    
    # return results
    return ekf_out

def interpolate_2d_lidar(
    lidar_datafile="lidar_data_2d.csv",
    imu_datafile="imu_data_2d.csv"):

    """
    Interpolate 2d LiDAR data at IMU rate
    """
    # load data
    imu_data = pd.read_csv(imu_datafile)
    imu_t_vec = imu_data["time"]

    lidar_data = pd.read_csv(lidar_datafile)
    lidar_t_vec = lidar_data["time"]
    px_data = lidar_data["r_measure"]*np.cos(lidar_data["theta_measure"])
    py_data = lidar_data["r_measure"]*np.sin(lidar_data["theta_measure"])

    px_itp = np.interp(imu_t_vec, lidar_t_vec, px_data)
    py_itp = np.interp(imu_t_vec, lidar_t_vec, py_data)

    # return results
    return {
        "px": px_itp,
        "py": py_itp,
        "time": imu_t_vec
    }


def test_2d_fusion_ekf():
    """
    test default settings for 2D Extended Kalman Filter that fuses Lidar + IMU
    """
    filter_data = run_2d_imu_lidar_ekf()
    ideal_data = calc_ideal_2d_pos()

    # plot results
    plt.figure()
    plt.plot(filter_data["px"], filter_data["py"], label="EKF")
    plt.plot(ideal_data["px"], ideal_data["py"], label="ideal")
    plt.legend()
    plt.title("EKF")

def compare_2d_results():
    """
    Compare Results for 2D problem
    """
    imu_int_data = run_2d_imu_kf()
    lidar_itp_data = interpolate_2d_lidar()
    fusion_data = run_2d_imu_lidar_ekf()
    ideal_data = calc_ideal_2d_pos()
    plt.figure()
    plt.plot(imu_int_data["px"], imu_int_data["py"],"--",label="IMU only", linewidth=2, color="blue")
    plt.plot(lidar_itp_data["px"], lidar_itp_data["py"],"-.",label="LiDAR only", linewidth=2, color="orange")
    plt.plot(fusion_data["px"], fusion_data["py"], label="Fusion",linewidth=2, color="green")
    plt.plot(ideal_data["px"], ideal_data["py"], label="Ideal",linewidth=2, color="black")
    plt.legend()
    plt.title("Motion Trajectory: 2D Algorithms Comparison")
    plt.xlabel("Y Position [m]")
    plt.ylabel("X Position [m]")
    plt.grid()
    plt.tight_layout()

    # calculate error
    itp_error = np.sqrt((lidar_itp_data["px"]-ideal_data["px"])**2 + (lidar_itp_data["py"]-ideal_data["py"])**2)
    imu_error = np.sqrt((imu_int_data["px"] - ideal_data["px"])**2+(imu_int_data["py"] - ideal_data["py"])**2)
    fusion_error = np.sqrt((fusion_data["px"] - ideal_data["px"])**2+(fusion_data["py"] - ideal_data["py"])**2)

    # print RMS distance error for each algorithm
    rms_itp = np.sqrt(sum(itp_error**2))
    rms_imu = np.sqrt(sum(imu_error**2))
    rms_fusion = np.sqrt(sum(fusion_error**2))

    print("RMS LIDAR: " + str(rms_itp))
    print("RMS IMU: " + str(rms_imu))
    print("RMS FUSION: " + str(rms_fusion))

    # plot results
    plt.figure()
    plt.plot(imu_int_data["time"], imu_error,"--",label="IMU only", linewidth=2, color="blue")
    plt.plot(lidar_itp_data["time"], itp_error,"-.",label="LiDAR only", linewidth=2, color="orange")
    plt.plot(fusion_data["time"], fusion_error, label="Fusion",linewidth=2, color="green")
    plt.legend()
    plt.title("Distance Error: 2D Algorithms Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance Error [m]")
    plt.grid()
    plt.tight_layout()


if __name__ == "__main__":
    test_imu_kf()
    test_1d_sensor_fusion_kf()
    compare_1d_results()
    test_imu_2d_kf()
    test_2d_fusion_ekf()
    compare_2d_results()
    plt.show()

