




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

    

def exp_distance(x,x_mainland,y_mainland):
    return [
            np.sum(np.exp(((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)*(x[0]-x_mainland)/((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2), 
            np.sum(np.exp(((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)*(x[1]-y_mainland)/((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**1/2)]





start_value = [0,0]

pol_dist = partial(exp_distance, x_mainland=x_mainland,y_mainland=y_mainland)
centroid_exp = fsolve(pol_dist,start_value)
    

plt.plot(country_border[0],country_border[1])
plt.scatter(centroid_exp[0],centroid_exp[1],c = "g")

plt.show()



