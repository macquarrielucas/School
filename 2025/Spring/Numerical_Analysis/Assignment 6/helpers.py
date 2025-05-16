import numpy as np
import logging as lg
import matplotlib.pyplot as plt
from numpy.linalg import norm, solve, cond
from tqdm import tqdm
from functions import h
from concurrent.futures import ProcessPoolExecutor
import datetime
lg.basicConfig(level=lg.INFO, format='%(levelname)s: %(message)s')
'''
Contains the information for a single run
of the modified Newton method.
'''
class Iteration:
    def __init__(self, x0, f, Df, epsilon, max_iters, suppress_logging = False):
        #suppress logging here is not good practice but doesn't matter.
        self.x0 = x0
        self.f = f
        self.Df = Df
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.iterates, self.errorsf, self.errorsxs = modified_newton(x0, f, Df, epsilon, max_iters, suppress_logging)
        self.length = len(self.iterates)
        self.initial_point = self.iterates[0]
        self.final_point = self.iterates[-1]


def plot_histogram(iterations_list):
    '''
    Takes a list of iterations and plots a histogram
    showing the length. This can be used to justify
    setting a maximum iteration length'''
    coordinates, iters = zip(*iterations_list)
    lengths = [iteration.length for iteration in iters]
    n_iters = len(iterations_list)
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=20, color='skyblue')
    plt.title(f'Histogram of Number of Iterations to Converge (n={n_iters})')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_color_map_squares(iterations_list, bounds, square_size=0.3):
    '''
    Takes a list of iterations and plots a color map showing the number of iterations to converge.
    Each point is represented as a circle with color and transparency (alpha) based on the number of iterations.
    Black circles indicate failure to converge within the maximum number of iterations.

    Parameters:
    - iterations_list: List of tuples containing coordinates and Iteration objects.
    - bounds: Tuple indicating the bounds of the grid (min_x, max_x, min_y, max_y).
    '''
    max_iters = max(iteration.length for iteration in iterations_list)
    min_x, max_x = bounds
    min_y, max_y = bounds

    fig, ax = plt.subplots(figsize=(8, 6))

    for iteration in iterations_list:
        x, y = iteration.x0
        length = iteration.length
        if length >= max_iters - 1:
            color = 'gainsboro'
            alpha = 1
        else:
            color = plt.cm.viridis(length / max_iters)
            alpha = 1
        square = plt.Rectangle((x, y), square_size, square_size, color=color, edgecolor=None, alpha=alpha)
        ax.add_artist(square)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title('Color Map of Iterations to Converge')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_iters))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Number of Iterations')

    plt.grid(False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"color_map_{timestamp}.png",dpi=600)

def _run_single_iteration_task(point, func, dfunc, eps, max_iters, suppress):
    '''Helper function to create and run an Iteration object for pool processing.'''
    return Iteration(point, func, dfunc, eps, max_iters, suppress)

def generate_iterations_processpool(f, Df, bounds=(-5, 5), grid_points=10, max_iters=100, epsilon=1e-6, suppress_logging = False):
    '''
    Generates a grid of initial points for the modified Newton method.
    Uses multiprocessing to speed up the process.
    Takes as input:
        - f: function to minimize
        - Df: derivative of f
        - bounds: bounds for the grid
        - grid_points: number of points in each dimension
        - max_iters: maximum number of iterations (default is 100)
        - epsilon: tolerance for convergence (default is 1e-6)
    '''
    x_vals = np.linspace(*bounds, grid_points)
    y_vals = np.linspace(*bounds, grid_points)
    init_points = [(xi, yj) for xi in x_vals for yj in y_vals]

        # Calculate the total number of points
    num_points = len(init_points)

    with ProcessPoolExecutor() as executor:
        # this needs to take lists as inputs, so make them length num_points
        iterations = list(executor.map(
            _run_single_iteration_task, 
            init_points,              
            [f] * num_points,         
            [Df] * num_points,        
            [epsilon] * num_points,   
            [max_iters] * num_points, 
            [suppress_logging] * num_points 
        ))

    return iterations # Return the list of Iteration objects

def generate_iterations(f, Df, bounds=(-5, 5), grid_points=10, max_iters=100, epsilon=1e-6, suppress_logging = False):
    '''
    Generates a grid of initial points for the modified Newton method.
    Takes as input:
        - f: function to minimize
        - Df: derivative of f
        - bounds: bounds for the grid
        - grid_points: number of points in each dimension
        - max_iters: maximum number of iterations (default is 100)
        - epsilon: tolerance for convergence (default is 1e-6)
    '''
    x_vals = np.linspace(*bounds, grid_points)
    y_vals = np.linspace(*bounds, grid_points)
    iterations = []

    total_points = len(x_vals) * len(y_vals)
    with tqdm(total=total_points, desc="Generating Iterations", unit="point") as pbar:
        for i, xi in enumerate(x_vals):
            for j, yj in enumerate(y_vals):
                lg.debug(f"Calculating initial points ({xi},{yj})")
                #The info from this can spam the output so we set it to critical 
                iteration = Iteration((xi, yj), f, Df, epsilon, max_iters, suppress_logging)
                lg.debug(f" Coord: ({xi},{yj}). Number of iterations: {iteration.length}")
                iterations.append(((i, j), iteration))
                pbar.update(1)

    return iterations

'''
My implementation of the modified Newton method. Only works for 2D functions.

Takes as input:
    - x0: initial guess
    - f: function to minimize
    - Df: derivative of f
    - epsilon: tolerance for convergence
    - max_iters: maximum number of iterations (default is infinity)
Returns:
    - iterates: list of iterates
    - errorsf: list of errors in function values
    - errorxs: list of errors in x values

'''
def modified_newton(x0, f, Df, epsilon, max_iters=np.inf, suppress_logging = False):
    xk = x0
    #Arrays to return the iterates and errors
    iterates = [x0]
    errorsf = [norm(f(*xk), ord=2)] #|f(xk)|_2
    errorxs = [np.inf] #|xk-xk-1|_2
    iteration = 0
    #initialize the error 
    error = np.inf
    while error > epsilon and iteration < max_iters:
        Dfk= Df(*xk)
        fk = f(*xk)
        
        try:
            dk = solve(Dfk, fk)
        except np.linalg.LinAlgError as e:
            if not suppress_logging:
                lg.error(f"LinAlgError: {e}")
                lg.info("Matrix is singular, stopping iteration.")
            break
        gammak = 1/cond(Dfk)
        Dhk =  np.array([[2*fk[0]*Dfk[0][0], 2*fk[1]*Dfk[0][1]],
                        [2*fk[0]*Dfk[1][0], 2*fk[1]*Dfk[1][1]]])
        def hk(t):
            return h(f, *(xk-t*dk))
        
        #Find smallest integer satisfying [...]
        j=0
        while  hk(2**(-j)) > hk(0) - (2**(-j))*gammak*norm(dk)*norm(Dhk, ord=2)/4:
            j+=1
        #List all the hk(2^-i) then find the minimum
        candidates = [] #each entry is (i, hk(2^-i))
        for i in range(0, j+1):
            candidates.append((i, hk(2**(-i))))
        candidates.sort(key=lambda x: x[1])
        i= candidates[0][0]
        lg.debug(f"Candidates: {candidates}")
        lambdak = 2**(-i)
        #Update xk
        xk_last = xk
        xk = xk-lambdak*dk
        #Add to the list of iterates
        iterates.append(xk)
        #Update tolerance
        errorf = norm(f(*xk), ord=2)
        errorx = norm(xk-xk_last, ord=2)
        errorsf.append(errorf)
        errorxs.append(errorx)
        #Update error for tolerance
        error = errorf 
        lg.debug(f"initial condition: {x0} \n step: {iteration } \n Iteration: {xk}, \n f(xk): {f(*xk)}, \n error: {error}")
        iteration += 1
    #Log the final step
    if not suppress_logging:
        lg.info(f"===FINAL RESULTS=== \n total steps: {iteration } \n Initial Condition: {x0} \n Iteration: {xk}, \n f(xk): {f(*xk)}, \n error: {error}")
    return iterates, errorsf, errorxs
# The output should be the final result of the optimization process, which is the point where the function f(x,y) is minimized.

def plot_2d(x, y, title='2D Plot', xlabel='X-axis', ylabel='Y-axis'):
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)  # Create a meshgrid for X and Y
    # Create a figure with two 3D subplots
    fig = plt.figure(figsize=(14, 6))

    def f1(x, y):
        return np.exp(x**2 + y**2) - 3

    def f2(x, y):
        return x + y - np.sin(3 * (x + y))


    # First subplot for the first component of f(x, y)
    ax1 = fig.add_subplot(121, projection='3d')
    Z1 = f1(X, Y)  # Apply f1 to the meshgrid
    zero = np.zeros_like(Z1)  # Create a zero array for the surface
    surf1 = ax1.plot_surface(X, Y, Z1, cmap=plt.cm.viridis, edgecolor='none')
    surf3 = ax1.plot_surface(X, Y, zero, cmap=plt.cm.viridis, edgecolor='none', alpha=0.5)
    ax1.set_title("First Component of f(x, y)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("f1(X, Y)")
    ax1.set_zlim(-10, 50)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Second subplot for the second component of f(x, y)
    ax2 = fig.add_subplot(122, projection='3d')
    Z2 = f2(X, Y)  # Apply f2 to the meshgrid
    surf2 = ax2.plot_surface(X, Y, Z2, cmap=plt.cm.plasma, edgecolor='none', alpha=0.5)
    surf4 = ax2.plot_surface(X, Y, zero, cmap=plt.cm.viridis, edgecolor='none', alpha=0.5)
    ax2.set_title("Second Component of f(x, y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("f2(X, Y)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()

def plot_convergence(iterates, errorsf, errorsxs):
    # 2D plot of iterates
    iterates_arr = np.array(iterates)

    fig, axs = plt.subplots(1,2,figsize=(8, 6))
    axs[0].plot(iterates_arr[:, 0], iterates_arr[:, 1], marker='o', linestyle='-', color='k', label='Iterates')
    axs[0].scatter(iterates_arr[0, 0], iterates_arr[0, 1], color='g', s=100, label='Start', zorder=5)
    axs[0].scatter(iterates_arr[-1, 0], iterates_arr[-1, 1], color='r', s=100, label='End', zorder=5)
    axs[0].set_title('Iterates in 2D')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend()
    axs[0].grid(True)
    #axs[1].grid(True)

    # Plot errorsf and errorsxs
    axs[1].semilogy(errorsf, marker='o', label='||f(x_k)||')
    axs[1].semilogy(errorsxs, marker='x', label='||x_k - x_{k-1}||')
    axs[1].set_title('Convergence of Modified Newton Method')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Error (log scale)')
    axs[1].legend()
    axs[1].grid(True, which='both')

    fig.tight_layout()