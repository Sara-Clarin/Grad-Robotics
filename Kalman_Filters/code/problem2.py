import matplotlib.pyplot as plt
import numpy as np
from math import sin


class KalmanFilter():
    """
    Implementation of a Kalman Filter.
    """
    def __init__(self, mu, sigma, A, C, R=0., Q=0., w=0.):
        """
        :param mu: prior mean
        :param sigma: prior covariance
        :param A: process model
        :param C: measurement model
        :param R: process noise
        :param Q: measurement noise
        """
        # prior
        self.mu = mu
        self.sigma = sigma
        self.mu_init = mu
        self.sigma_init = sigma
        # process model
        self.A = A
        self.R = R
        # measurement model
        self.C = C
        self.Q = Q
        self.v = 0
        self.w = w

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
        z_t = sensor_data
        
        # initialize accumulators with 0s
        mu_total = np.array(np.zeros((100,4,1)))
        sig_total = np.array(np.zeros(( 100,4,4)))

        for i in range(z_t.shape[0]):  # run predictions for t=timestep trials
            self._predict_4()
            self._update_4(z_t[i])
            mu_total[i] = self.mu + self.w[i]
            sig_total[i] = self.sigma

        return (mu_total, sig_total)

    
    def _predict_4(self):    # first two steps of Kalman filter algorithm
        self.mu = np.dot(self.A, self.mu)
        self.sigma = np.dot(np.dot(self.A, self.sigma), self.A.T) + self.R

    def _update_4(self,z):   # rest of kalman filter algorithm
        y = z - np.dot(self.C, self.mu)
        S = self.Q + np.dot(self.C, np.dot(self.sigma, self.C.T))
    
        K = np.dot(np.dot(self.sigma, self.C.T), np.linalg.inv(S))
        self.mu = self.mu + np.dot(K, y)
        I = np.eye(4)
        self.sigma = np.dot(I - np.dot(K, self.C), self.sigma)


def plot_prediction(t, ground_truth, measurement, predict_mean, predict_cov):
    """
    Plot ground truth vs. predicted value.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param measurement: Tx1 array of sensor values
    :param predict_mean: TxD array of mean vectors
    :param predict_cov: TxDxD array of covariance matrices
    """
    predict_pos_mean = predict_mean[:, 0].squeeze()
    predict_pos_std = predict_cov[:, 0, 0]
    plt.figure()
    plt.plot(t, ground_truth, color='k')
    plt.plot(t, measurement, color='r')
    plt.plot(t, predict_pos_mean, color='g')
    
    plt.fill_between(
        t,
        predict_pos_mean-predict_pos_std,
        predict_pos_mean+predict_pos_std,
        color='g',
        alpha=0.5)
    plt.legend(("ground truth", "measurements", "predictions"))
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("Predicted Values")
    plt.show()

def plot_mse(t, ground_truth, predict_means):
    """
    Plot MSE of your KF over many trials.

    :param t: 1-dimensional array representing timesteps, in seconds.
    :param ground_truth: Tx1 array of ground truth values
    :param predict_means: NxTxD array of T mean vectors over N trials
    """
    predict_pos_means = predict_means[:, :, 0]
    print(f'predict_pos_means shape is: {predict_pos_means.shape}')
    print(f'ground_truth shape is: {ground_truth.shape}')
    errors = ground_truth.squeeze() - predict_pos_means
    mse = np.mean(errors, axis=0) ** 2
    print(f'mse shape is: {mse.shape}')
    print(f't is: {t}')

    plt.figure()
    plt.plot(t , mse/1.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Position MSE (m^2)")
    plt.title("Prediction Mean-Squared Error")
    plt.show()
    
def problem2a( mu, sigma, A, C, t):
    #Initially, we have noise R = 0
    t = 100
    R = np.zeros(4)  # part A: we have 0 matrix process noise
    Q = np.array([1]).reshape(1,1)

    # create noisy data with shape sin(0.1*t)
    X = np.linspace(0,100,100)
    zt = np.sin( 0.1*X) + np.random.normal(0,1,100)
    w = np.random.normal(0,0.1,100)
    
    input_t = np.linspace(0,100,100)
    ground_truth = np.sin(0.1*input_t)

    kalman = KalmanFilter(mu,sigma, A, C, R, Q, w)
    predict_mean, predict_cov = kalman.run( zt )
  
    plot_prediction(X, ground_truth, zt, predict_mean, predict_cov)

    # plot MSE with 1000 trials
    n_trials = 1000
    pred_means  = np.array(np.zeros((n_trials,t ,4)))
    for i in range(n_trials):
       kalman.reset()
       predict_mean, predict_cov = kalman.run(zt)
       pred_means[i] = predict_mean.squeeze()

    plot_mse( input_t, ground_truth.reshape(100,1) , pred_means)

def problem2b( mu, sigma, A, C, t):
    # part B: fictitious noise
    R = np.eye(4) * 0.1  # DIFFERENT from part A: add 'process noise'
    t = 100
    Q = np.array([1]).reshape(1,1)

    # create noisy data with shape sin(0.1*t)
    X = np.linspace(0,100,100)
    zt = np.sin( 0.1*X) + np.random.normal(0,1,100)
    w = np.random.normal(0,0.1,100)
    
    input_t = np.linspace(0,100,100)
    ground_truth = np.sin(0.1*input_t)

    kalman = KalmanFilter(mu,sigma, A, C, R, Q, w)
    predict_mean, predict_cov = kalman.run( zt )
  
    plot_prediction(X, ground_truth, zt, predict_mean, predict_cov)

    # Plot MSE with 1000 trials
    n_trials = 1000
    pred_means  = np.array(np.zeros((n_trials,t ,4)))
    for i in range(n_trials):
       kalman.reset()
       predict_mean, predict_cov = kalman.run(zt)
       pred_means[i] = predict_mean.squeeze()

    plot_mse( input_t, ground_truth.reshape(100,1) , pred_means)

if __name__ == '__main__':

    mu = np.array([5, 1, 0, 0]).reshape(4,1)
    sigma = np.array([[10, 0, 0, 0],[0, 10, 0, 0],[0, 0, 10, 0],[0, 0, 0, 10]])
    A = np.array([[1, 0.1, 0, 0],  [0, 1, 0.1, 0 ], [0, 0, 1, 0.1], [0,0,0,1]])
    C = np.array([1, 0, 0, 0]).reshape(1,4)
    
    problem2a(mu, sigma, A, C, t=100)
    problem2b(mu, sigma, A, C, t=100)
