from functools import wraps
import time


def time_measurement(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"{func.__name__} took {elapsed_time} seconds.")  # Print function name and elapsed time
        return result
    return wrapper