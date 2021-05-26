import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set()

def calc_trajectory(theta, v0, Cd = 0.6, den_air = 1.3, dt=0.1, n_iter = 10000, plot= False):
    g = 9.81
    area_ball = 0.00403
    mass = 0.16
    #initialise lists
    ax = np.zeros(n_iter)
    ay = np.zeros(n_iter)
    vx = np.zeros(n_iter)
    vy = np.zeros(n_iter)
    x = np.zeros(n_iter)
    y = np.zeros(n_iter)
    times = np.zeros(n_iter)

    vx[0] = v0 * np.cos(theta * (2*np.pi)/360)
    vy[0] = v0 * np.sin(theta * (2*np.pi)/360)

    x[0] = 0
    y[0] = 0
    final_iter = 0
    for i in range(n_iter -1):
        dragy = 1/2 * Cd * den_air * area_ball * (vy[i])**2
        dragx = 1/2 * Cd * den_air * area_ball * (vx[i])**2

        accy = -(np.sign(vy[i])*dragy)/mass - g 
        accx = -(np.sign(vx[i])*dragx)/mass

        vy[i+1] = vy[i] + accy * dt
        vx[i+1] = vx[i] + accx * dt

        if y[i] < 0:
            final_iter = i
            break
        y[i+1] = y[i] + vy[i+1] *dt
        x[i+1] = x[i] + vx[i+1] *dt
    if plot == True:
        plt.plot(x[:final_iter] , y[:final_iter])
    return x[final_iter]



best_angles = []
velocities = np.arange(10, 60,1 )
for v in velocities:
    xvals = []
    angles = []
    for i in np.arange(0, 90, 0.1):
        x_dist = calc_trajectory(i,v, dt=0.05, n_iter= 100000)
        xvals.append(x_dist)
        angles.append(i)
    best_angles.append(angles[np.argmax(xvals)])
plt.show()

plt.plot(velocities, best_angles)
plt.xlabel('bat exit velocity')
plt.ylabel('optimum trajectory angle')
    


