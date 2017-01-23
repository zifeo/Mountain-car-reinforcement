import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from mountaincar import MountainCar, MountainCarViewer

np.seterr(over='raise')

car = MountainCar()

### Helper coordinates

# grid discretisation
ngrid_pos = 20
ngrid_speed = 20

# grid corners
int_pos = -150, 30
int_speed = -15, 15

# prepare all parameters combination for states
x_pos, center_dist_pos = np.linspace(int_pos[0], int_pos[1], ngrid_pos, retstep=True)
y_speed, center_dist_speed = np.linspace(int_speed[0], int_speed[1], ngrid_speed, retstep=True)
y_speed_t = y_speed.reshape(-1, 1)


# ### Helper functions

def activity(s):
    '''Given a state `s`, returns local continuous activity response.'''
    return np.exp(- ((x_pos - s[0]) / center_dist_pos) ** 2 - ((y_speed_t - s[1]) / center_dist_speed) ** 2).T

def Q(s, a, w):
    '''Given a state `s`, an action `a` and weights `w`, returns corresponding q-values.'''
    return np.sum(w[:, :, a] * activity(s))

def softmax(x, tau):
    '''Given an array `x` and a temperature parameter `tau`, returns robust softmax applied on the array.'''
    delta = (np.max(x) - np.min(x))
    
    # all zero mean 1/len(x) chance for each action
    if np.isclose(delta, 0):
        return np.ones_like(x) / len(x)
    
    # rescale to avoid overflow issues
    xp = (np.array(x) - np.min(x)) / delta
    
    e_x = np.exp(xp / tau)
    return e_x / e_x.sum()

### SARSA

def sarsa(n_epi = 100,
          tau_max=40, # exploration/expoitation parameter
          tau_min=1e-2,
          gamma=0.95, 
          lmbda = 0.95, 
          eta = 0.01,
          fill = 0,
          show = False,
          limit = 2000,
          dt=0.01, 
          steps=100):
    '''Given hyperparameters, run the sarsa algorithm on `n_epi` episode and returns weight, probabilities and latency history.'''
    
    # store latency and probabilities
    probs = []
    latencies = []

    # decreasing exponential coeficient for tau
    tau_coef = np.exp((1 / n_epi) * np.log(tau_min / tau_max))
    tau = tau_max
    
    # initial random weights (at least connected)
    #w = np.random.rand(ngrid_pos, ngrid_speed, 3) / 10. + 1
    w = np.ones((ngrid_pos, ngrid_speed, 3)) * fill

    for epi in np.arange(n_epi):
        print("------------------------")
        print("episode :", epi)
        print("with tau :", tau)

        # null eligibility traces
        e = np.zeros((ngrid_pos, ngrid_speed, 3))

        # initial state
        car.reset()
        s0 = car.x, car.x_d

        # initial random action
        a0 = np.random.randint(3)

        j = 0
        while j < limit:
            j += 1

            # take action, simulate and retrieve new state
            car.apply_force(a0 - 1)
            car.simulate_timesteps(steps, dt)
            s1 = car.x, car.x_d

            # compute probabilities for each action and choose among them
            p = softmax([Q(s1, a, w) for a in range(3)], tau)
            a1 = np.random.choice(range(3), p=p)

            # decrease eligibility traces and increase selected action
            e *= gamma * lmbda
            e[:, :, a0] += activity(s0)[:, :]

            # update weights accordingly
            # the factor j / 1000 has been added after discussion with TAs in order to increase convergence speed
            delta = car.R + gamma * Q(s1, a1, w) - Q(s0, a0, w) - j / 1000.
            w += eta * delta * e

            # propagate next action and state
            a0 = a1
            s0 = s1

            if car.R > 0.0:
                print('reward obtained at t =', car.t)
                break
                
            # tau update (minimum value to prevent overflow)
            #tau = max(tau * tau_coef, tau_min)
            #print("new tau", tau)
        
        tau *= tau_coef
        
        prob = np.array([[softmax([Q((x, y), a, w) for a in range(3)], tau) for x in x_pos] for y in y_speed])
        max_action = -np.max([[[Q((x, y), a, w) for a in range(3)] for x in x_pos] for y in y_speed], axis=2)

        if (show):
            vec_plot(prob)
            plot3D(max_action)
            plt.show()

        probs.append(prob)
        latencies.append(car.t)
        
    return w, probs, latencies


# In[48]:

_id = sys.argv[1]
def run(tau, lmbda, fill, t_reduce=False):
    
    if not t_reduce:
        text_add = ""
        w, probs, latencies = sarsa(tau_min=tau, tau_max=tau, 
                                    lmbda=lmbda, fill=fill)
    else:
        text_add = "_tau_reduce"
        w, probs, latencies = sarsa(tau_min=0.1, tau_max=40, 
                                    lmbda=lmbda, fill=fill)
        
        
    data = { "probabitity": probs, "latencies": latencies}
    df = pd.DataFrame(data)
    
    df.to_pickle("./result_" + str(_id) + "_tau_" + str(tau) + "_lmbda_" + 
              str(lmbda) + "_fill_" + str(fill) + text_add)


for tau in [0, *np.linspace(0.1, 100, 20)]:
    for lmbda in np.linspace(0, 0.95, 20):
        for fill in [0,1]:
            if tau == 0:
                run(tau, lmbda, fill, True)
            else:
                run(tau, lmbda, fill, False)

