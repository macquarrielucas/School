import time
import logging as lg
import pickle
from helpers import (
    modified_newton,
    generate_iterations,
    generate_iterations_processpool,
    plot_color_map_squares # noqa: F403
)
from functions import f1, Df1 #Assignent functions
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning) #This is to suppress the overflow warnings from exp(x^2+y^2)

# Add color to logging
class CustomFormatter(lg.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m' # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)

# Configure logging with color
handler = lg.StreamHandler()
handler.setFormatter(CustomFormatter('%(levelname)s: %(message)s'))
lg.getLogger().handlers = [handler]

def main():
    lg.info("=====Starting main method=====")
    epsilon = 1e-6
    bounds=(-10,10) 
    grid_points=500
    max_iters=150
    epsilon=epsilon
    cache_file = f"iterations_cache_{grid_points}x{grid_points}_maxitters_{max_iters}_bounds_{bounds[0]}_{bounds[1]}.pkl"
    #cache_file = "/home/lucas/Documents/School/2025/Spring/Numerical_Analysis/Assignment 6/iterations_cache_500x500_maxitters_50_bounds_-4.5_4.5.pkl"
    lg.info(f"Searching for cache file: {cache_file}")
    try:
        # Try to load cached iterations
        with open(cache_file, "rb") as f:
            iterations = pickle.load(f)
        lg.info("Loaded iterations from cache.")
    except (FileNotFoundError, EOFError):
        # If cache doesn't exist or is invalid, generate iterations
        lg.info("Cache not found. Generating iterations...")
        start_time = time.time()
        iterations = generate_iterations_processpool(f1, 
                        Df1, 
                        bounds=bounds, 
                        grid_points=grid_points,
                        max_iters=max_iters, 
                        epsilon=epsilon,
                        suppress_logging=True)
        end_time = time.time()
        lg.info(f"Time taken to generate iterations: {end_time - start_time:.2f} seconds")

        # Save iterations to cache
        with open(cache_file, "wb") as f:
            pickle.dump(iterations, f)
        lg.info("Iterations saved to cache.")

    plot_color_map_squares(iterations, bounds=bounds, square_size=0.075)

    lg.info("=====Ending main method=====")


if __name__ == '__main__':
    main()