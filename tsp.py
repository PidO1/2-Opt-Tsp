import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import math  
import sys
import time

from matplotlib import animation
# matplotlib.use("Agg")

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


# plt.gca().invert_yaxis()
xs = []
ys = []
route1 = []
distance_mat = []
optimalR=[]
names=[]
namesInOrder=[]

# Calculate the euclidian distance
path_distance = lambda r,c: np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))])
# Reverse
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(cities,improvement_threshold): 
    route = np.arange(cities.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    print(route)
    print("next")
    print(best_distance)
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                distance_mat.append(new_distance)
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
    return route # When the route is no longer improving substantially, stop searching and return the route.
# Create a matrix of cities, with each row being a location in 2-space (function works in n-dimensions).
def generate():
        # xs = np.random.randint(self.width, size=self.nodesNumber)
        # ys = np.random.randint(self.height, size=self.nodesNumber)
        with open('Locations.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                # print(f'\t{row[0]} ***** {row[1]} ***** {row[2]}.')
                    xs.append(float(row[2]))
                    ys.append(float(row[1]))
                    names.append(row[0])

                    line_count += 1                
                
                
                    

        return np.column_stack((xs, ys))
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

# def calculateDistance(route,i):  
#     # print(i)
#     # print(route)
#     dist=0
#     idx = 1
#     while idx < len(new_cities_order)-1:
#         dist += math.sqrt( ((new_cities_order[idx+1][0])-(new_cities_order[idx][0]))**2 +   ((new_cities_order[idx+1][1])-(new_cities_order[idx][1]))**2)
#         idx+=1
#     return dist
    
    
    
cities = generate()
# Find a good route with 2-opt ("route" gives the order in which to travel to each city by row number.)
route = two_opt(cities,0.001)
# print(names)

new_cities_order = np.concatenate((np.array([cities[route[i]] for i in range(len(route))]),np.array([cities[0]])))
print(new_cities_order)

xstartdata = new_cities_order[0][0]
ystartdata = new_cities_order[0][1]
fig, ax = plt.subplots()


# plt.grid(True)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
ln, = plt.plot([], [], '--bo', animated=True)
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.plot([xstartdata], [ystartdata], '--bo',  lw=5)
x = []
y = []
for i, txt in enumerate(names):
    ax.annotate(txt, (cities[i][0],cities[i][1]))
extra_x = (max(xs) - min(xs)) * 0.05
extra_y = (max(ys) - min(ys)) * 0.05
ax.set_xlim(min(xs) - extra_x, max(xs) + extra_x)
ax.set_ylim(min(ys) - extra_y, max(ys) + extra_y)
def init():
    ln.set_data([],[])
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-100, 100)

    ax.set_title('Frame ')
    

    return ln,
def animate(i):
    x.append(new_cities_order[i][0])
    y.append(new_cities_order[i][1])
    ln.set_data(x, y)  
    return ln,


anim = animation.FuncAnimation(fig, animate, init_func=init,frames=range( len(new_cities_order)), interval=100,blit=True,repeat=False)

plt.show()





