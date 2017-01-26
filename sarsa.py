
# coding: utf-8

# In[1]:



# In[2]:

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D


# In[3]:

from mountaincar import MountainCar, MountainCarViewer


# In[4]:

# stop if any overflow encountered
np.seterr(over='raise');


# In[5]:

car = MountainCar()


# ### Plot functions

# In[6]:

def vec_plot(p):
    '''Give an array `p` of probabilities, plots the Q-values direction vector field.'''
    p_max = np.argmax(p, axis=2)
    
    # define arrow direction
    U = p_max - 1
    V = np.zeros((ngrid_pos, ngrid_speed))

    plt.quiver(U, V, alpha=1, scale=1.8, units='xy')
    plt.xlim(-1, 20)
    plt.xticks(())
    plt.ylim(-1, 20)
    plt.yticks(())
    plt.xlabel('position $x$')
    plt.ylabel('speed $\dot x$')
    plt.title('Q-values direction vector field (arrows show the direction of applied force)')

    plt.show()


# In[7]:

def plot3D(q):
    '''Given q-values `q`, plots in 3D all possibles states.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # generate all parameters combination for states
    x, y = np.meshgrid(x_pos, y_speed)
    ax.plot_wireframe(x, y, q, color='grey')
    ax.set_xlabel('position')
    ax.set_ylabel('speed')
    ax.set_zlabel('max q')
    
    plt.show()


# ### Helper coordinates

# In[8]:

# grid discretisation
ngrid_pos = 20
ngrid_speed = 20


# In[9]:

# grid corners
int_pos = -150, 30
int_speed = -15, 15


# In[10]:

# prepare all parameters combination for states
x_pos, center_dist_pos = np.linspace(int_pos[0], int_pos[1], ngrid_pos, retstep=True)
y_speed, center_dist_speed = np.linspace(int_speed[0], int_speed[1], ngrid_speed, retstep=True)
y_speed_t = y_speed.reshape(-1, 1)


# ### Helper functions

# In[11]:

def activity(s):
    '''Given a state `s`, returns local continuous activity response.'''
    return np.exp(- ((x_pos - s[0]) / center_dist_pos) ** 2 - ((y_speed_t - s[1]) / center_dist_speed) ** 2).T


# In[12]:

def Q(s, a, w):
    '''Given a state `s`, an action `a` and weights `w`, returns corresponding q-values.'''
    return np.sum(w[:, :, a] * activity(s))


# In[13]:

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


# In[14]:

qs = np.array([0.3, 0.2, 0.1])
x = np.logspace(-1, 3, 40)
y = np.array(list(zip(*[softmax(qs, xi) for xi in x])))

plt.xscale("log")
plt.xlabel("tau")
plt.ylabel("probability")
plt.plot(x,y[0])
plt.plot(x,y[1])
plt.plot(x,y[2])
plt.show()


# In[15]:

tau_max = 5
tau_min = 1e-1
tau_steps = 100
x = range(tau_steps)
y = [tau_max * np.exp((1 / tau_steps) * np.log(tau_min / tau_max)) ** i for i in x]
plt.plot(x,y)


# ## SARSA algo

# In[16]:

def sarsa(n_epi=100,
          tau_max=1, # exploration/expoitation parameter
          tau_min=1e-1,
          expire=100,
          gamma=0.95, 
          lmbda = 0.05, 
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
    tau_coef = np.exp((1 / expire) * np.log(tau_min / tau_max))
    tau = tau_max
    
    # initial weights
    w = np.ones((ngrid_pos, ngrid_speed, 3)) * fill

    i = 0
    while i < n_epi:
        print("------------------------")
        print("episode :", i)
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
            delta = car.R + gamma * Q(s1, a1, w) - Q(s0, a0, w) - j / 1000
            w += eta * delta * e

            # propagate next action and state
            a0 = a1
            s0 = s1

            if car.R > 0.0:
                i += 1
                
                tau *= tau_coef
        
                prob = np.array([[softmax([Q((x, y), a, w) for a in range(3)], tau) for x in x_pos] for y in y_speed])
                max_action = -np.max([[[Q((x, y), a, w) for a in range(3)] for x in x_pos] for y in y_speed], axis=2)

                if (show):
                    vec_plot(prob)
                    plot3D(max_action)
                    plt.show()

                probs.append(prob)
                latencies.append(car.t)
                
                print('reward obtained at t =', car.t)
                break
        
    return w, probs, latencies


# In[17]:

w, probs, latencies = sarsa(show=True)


# ## Grid search

# The following code is used for finding best params and therefore is not required to be reviewed.

# In[ ]:

origin_path = "../grid_search/"
dfs = []

for subd, dirs, files in os.walk(origin_path):
    if len(dirs) > 0:
        continue
        
    folder = subd + "/"
    id_ = int(subd[-1])
    
    print("id :",id_)  
    
    links = []
    
    for file in files:
        if file == "log.txt":
            continue
        
        file_arr = file.split("_")
        
        if len(file_arr) > 8:
            tau = "reduce"
        else:
            tau = file_arr[3]
        
        lmbda =  file_arr[5]
        fill = file_arr[7]
        
        links.append({
            "tau": tau,
            "lmbda": float(lmbda),
            "fill": int(fill),
            "link": folder + file
        })
    
    dfs.append(links)


# In[ ]:

def search_run(obj):
    return  (obj["fill"] == search["fill"] and 
             obj["lmbda"] == search["lmbda"] and 
             obj["tau"] == search["tau"])


# In[ ]:

# imported from https://tonysyu.github.io/plotting-error-bars.html#.WIi79RiZOsw
def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    ax.set_xlabel("episodes")
    ax.set_ylabel("iterations to goal")
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


# In[ ]:

def plot_latencies(search_obj, a=None,smooth=10):
    if a is None:
        search = search_obj
        latencies = np.array(
            [
                pd.read_pickle(
                    list(filter(search_run, sims))[0]["link"]
                ).latencies for sims in dfs])
    else:
        latencies = np.array(a)
    lat_m = latencies.mean(axis=0)
    var = np.sqrt(latencies.var(axis=0))
    
    print("mean :", lat_m.mean())
    print("mean last 60 :", lat_m[60:].mean())
    print("min :", latencies.min())
    
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(100*[lat_m[60:].mean()], "red")
    errorfill(range(100), lat_m, var, color="gray")
    plt.show()


# In[ ]:

a_ = []
for i in range(8):
    w, probs, latencies = sarsa(fill=0, lmbda=0.05, tau_max=1, tau_min=1)
    a_.append(latencies)


# In[ ]:

a2 = []
for i in range(8):
    w, probs, latencies = sarsa(fill=0, lmbda=0.05, tau_max=1, tau_min=0.1)
    a2.append(latencies)


# In[ ]:

search = {"fill": 0, "lmbda": 0.05, "tau": '0.1'}
   
res = filter(search_run, dfs[0])

print(list(res))

plot_latencies(search, a=a2)


# In[ ]:

w, probs, latencies = sarsa(fill=0, lmbda=0.95, tau_max=10, tau_min=0.1, show=True)


# In[ ]:

w, probs, latencies = sarsa(fill=1, lmbda=0.95, tau_max=10, tau_min=0.1, show=False)


# In[ ]:

plt.plot(latencies)
plt.show()


# In[ ]:

lats = []

for i in range(20):
    w, probs, latencies = sarsa(fill=0, lmbda=0.05, tau_max=10, tau_min=0.1, show=False)
    lats.append(latencies)


# In[ ]:

lats = np.array(lats)

plt.figure(figsize=(10, 8), dpi=80)

for lat in lats:
    plt.plot(lat, "b", alpha=0.2)

lat_m = lats.mean(axis=0)
var = np.sqrt(lats.var(axis=0))

plt.plot(100*[lat_m[60:].mean()], "red", alpha=1)
errorfill(range(100), lat_m, var, color="black")
    
plt.show()


# In[ ]:

i = 0
latencies = np.array([pd.read_pickle(sims[i]["link"]).latencies for sims in dfs])


# In[ ]:

lat_m = latencies.mean(axis=0)
var = np.sqrt(latencies.var(axis=0))


# In[ ]:

# plot latency evolution and moving avergage of it
smooth = 10
plt.plot(lat_m, "b")
plt.plot(lat_m - var, "r+")
plt.plot(lat_m + var, "r+")
plt.plot(np.convolve(np.ones(smooth) / smooth, lat_m, mode='same'))

