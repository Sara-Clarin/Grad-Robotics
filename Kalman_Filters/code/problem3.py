import matplotlib.pyplot as plt
import numpy as np
import math


class ExtendedKalmanFilter():
    """
    Implementation of an Extended Kalman Filter.
    """
    def __init__(self, mu, sigma, g, g_jac, h, h_jac, R=0., Q=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param g: process function
        :param g_jac: process function's jacobian
        :param h: measurement function
        :param h_jac: measurement function's jacobian
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.g = g
        self.g_jac = g_jac
        self.R = R
        # measurement model
        self.h = h
        self.h_jac = h_jac
        self.Q = Q


    def reset(self):
        """
        Reset belief state to initial value.
        """
        self.mu = self.mu_init
        self.sigma = self.sigma_init

    def run(self, sensor_data):
        """
        Run the Kalman Filter using the given sensor updates.

        :param sensor_data: array of T sensor updates as a TxS array.

        :returns: A tuple of predicted means (as a TxD array) and predicted
                  covariances (as a TxDxD array) representing the KF's belief
                  state AFTER each predict/update cycle, over T timesteps.
        """
        # FILL in your code here
        d = sensor_data.shape[1]
        t = sensor_data.shape[0]

        mu_total = np.array(np.zeros((t,d, 1)))
        sig_total = np.array(np.zeros((t,d, d)))

        for i,z in enumerate(sensor_data):

            print(f'z is: {z}')
            print(f'--------------------------------')
            print(f'Time {i}')
            print(f'--------------------------------')
            self._predict()
            self._update(z)
            
            mu_total[i] = self.mu.reshape(2,1)
            sig_total[i] = self.sigma

        return (mu_total, sig_total)

    def _predict(self):
        print(f'Update phase: ')
        self.mu = g(self.mu)
        print(f'self.mu: {self.mu} has shape {self.mu.shape}')
        G = self.g_jac(self.mu)
        print(f'G is :{G} has shape {G.shape}')
        self.sigma = np.dot(G, self.sigma).dot(G.T) + ( np.eye(2) * (self.R))
        print(f'self.sig: {self.sigma} has shape {self.sigma.shape}')
        
    def _update(self, z):
        H = self.h_jac(self.mu)
        S = np.dot(H, self.sigma).dot(H) + self.Q # might get transposing issues here
        K = np.dot(self.sigma, H).dot( 1/S )
               
 
        y = z -  self.h(self.mu)
        self.mu = self.mu + K.dot(y)
        I = np.eye(2)
        self.sigma = ( I - K.dot(H)).dot(self.sigma)

        print(f'self.mu: {self.mu} has shape {self.mu.shape}')
        print(f'self.sigma: {self.sigma} has shape {self.sigma.shape}')

def plot_prediction(t,  predict_mean, predict_cov, gt_x, gt_a):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values  --> this is wrong as per piazza
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """

    pred_x, pred_a = predict_mean[:, 0].squeeze(), predict_mean[:, 1].squeeze()
    pred_x_std = np.sqrt(predict_cov[:, 0, 0]) 
    pred_a_std = np.sqrt(predict_cov[:, 1, 1])

    plt.figure(figsize=(7, 10))
    plt.subplot(211)
    plt.plot(t, gt_x, color='k')
    plt.plot(t, pred_x, color='g')
   
    print(f't: {t} \n pred_x_std: {pred_x_std} \n pred_a_std: {pred_a_std}')
    plt.fill_between(
        t,
        pred_x-pred_x_std,
        pred_x+pred_x_std,
        color='g',
        alpha=0.5)
    
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$x$")
    plt.title(r"EKF estimation: $x$")

    plt.subplot(212)
    plt.plot(t, gt_a, color='k')
    plt.plot(t, pred_a, color='g')
   
    ''' 
    plt.fill_between(
        t,
        pred_a-pred_a_std,
        pred_a+pred_a_std,
        color='g',
        alpha=0.5)
    '''
    plt.legend(("ground_truth", "prediction"))
    plt.xlabel("time (s)")
    plt.ylabel(r"$\alpha$")
    plt.title(r"EKF estimation: $\alpha$")

    plt.show()

def g(mu):
    x = mu[0]
    alpha = mu[1]
    return np.array( [x*alpha, alpha] )

def g_jac(mu):
    x = mu[0]
    alpha = mu[1]
    return np.array([x, alpha])

def h_jac(x):
    x1 = x[0] / (math.sqrt(x[0]**2 + 2)) 
    return np.array([x1, 0])

def h(x):
    x1 = math.sqrt(x[0]**2 + 1) # X0 is only x
    return np.array([x1, x[0]]) 

def h_1d(x):
    return math.sqrt(x**2 + 1)

def problem3():
    # FILL in your code here
    mu = np.array([1,2])   # [x, alpha]
    sigma = np.eye(2) 

    Q = 1
    R = 0.5
    alpha = 0.1

    # timestep
    T = 20 
    
    timesteps = np.linspace(0, T-1, T)
    w = np.random.normal(0, math.sqrt(R), T) # process noise
    v = np.random.normal(0, math.sqrt(Q), T) # measurement noise
    
    x = np.ones((T))
    alpha_const_v = x * alpha
    x[0] = 2
    
    for i in range(1,T):  # ground truth
        x[i] = alpha*x[i-1]

    x_a_gt = np.vstack((x, alpha_const_v) ).T
 
    zt = np.array([ h_1d(i) + w[i] for i in range(T)])

    EKF =  ExtendedKalmanFilter(mu, sigma, g, g_jac, h, h_jac, R=R, Q=Q )
    pred_mean, pred_cov = EKF.run(x_a_gt)
    plot_prediction(timesteps, pred_mean, pred_cov, x, alpha_const_v)

if __name__ == '__main__':
    problem3()
