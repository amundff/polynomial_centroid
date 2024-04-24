




import numpy as np
from functools import partial
from scipy.optimize import newton_krylov, fsolve
import matplotlib.pyplot as plt
input_file = np.load("saved_arrays/norway_grid_files.npz", allow_pickle=True)

    


grid_points = input_file["grid_points"].T
mainland_grid_points = input_file["mainland_grid_points"]
variables = input_file["variables"]
country_border = input_file["country_border"].T
num_intervals, buffer = variables
centroid = np.array([13.96066889901151,64.21675150208692])

mainland_grid_points -= centroid
mainland_max = np.max(mainland_grid_points)
country_border-= centroid
mainland_grid_points = mainland_grid_points.T
country_border = country_border.T
mainland_grid_points /= mainland_max
country_border /= mainland_max

x_mainland, y_mainland = mainland_grid_points

x_mainland = x_mainland.astype(np.longdouble)
y_mainland = y_mainland.astype(np.longdouble)

    

def sine_distance(x,x_mainland,y_mainland):
    return [
            np.sum(np.cos(((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)*(x[0]-x_mainland)/((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2), 
            np.sum(np.cos(((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)*(x[1]-y_mainland)/((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)]


a = np.linspace(-.4,.4,10)
b = np.linspace(-.6,1,10)
c = np.array(np.meshgrid(b,a))

e = np.vstack((c[0].ravel(), c[1].ravel())).T




start_value = [0,0]
d = np.zeros((100,2))
for i,j in enumerate(e):
    start_value = j
    pol_dist = partial(sine_distance, x_mainland=x_mainland,y_mainland=y_mainland)
    centroid_sine = fsolve(pol_dist,start_value)
    d[i] = centroid_sine


d = d.T


plt.plot(country_border[0],country_border[1])
# plt.scatter(c[0],c[1])
plt.scatter(d[0],d[1],c = "g")
plt.show()



