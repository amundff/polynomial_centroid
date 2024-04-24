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

def polynomial_distance(x, x_mainland, y_mainland, n) -> np.ndarray:
    
    return [
            np.sum(n/2* ((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**((n-2)/2) * 2*(x[0]-x_mainland)),
            np.sum(n/2* ((x[0]-x_mainland)**2 + (x[1] - y_mainland)**2)**((n-2)/2) * 2*(x[1]-y_mainland))]


    
start = 2
end = 2000
num_points = 1000

start_value = [17,65]-centroid
input_n = np.linspace(start,end,num_points,dtype=np.longdouble)
centroid_deg_n = np.zeros((num_points,2),dtype=np.longdouble) 
for index, value in enumerate(input_n):
    
    pol_dist = partial(polynomial_distance, x_mainland=x_mainland,y_mainland=y_mainland,n=value)
    centroid_deg = fsolve(pol_dist,start_value)
    centroid_deg_n[index] = centroid_deg
    start_value = centroid_deg
    
    
centroid_deg_n = centroid_deg_n.T

plt.plot(country_border[0],country_border[1])
plt.plot([-.4,1],[-.36,.366])
plt.scatter(centroid_deg_n[0],centroid_deg_n[1],s = 5,c="r")
plt.show()
