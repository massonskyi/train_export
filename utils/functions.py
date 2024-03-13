import os
import time
from roboflow import Roboflow

from config import ROBOFLOW_API_KEY
import logging
from functools import wraps

def __timer__(func):
    """
    Decorator to time a function call before returning the result of the function call.
    :param func: The function to time. This function must take no arguments. For example
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Function '{func.__name__}' executed in {elapsed_time_ms:.2f} ms")
        return result
    return wrapper

@__timer__
def configure_logging(log_file='app.log', log_level=logging.INFO, format_str=None):
    """
    Configure the logger and save the log file if it doesn't exist already and create it if it doesn't exist already.
    :param log_file: The log file name. If not specified it will default to 'app.log'
    :param log_level: The logging level to use. Can be 'debug', 'info', 'warning', 'error', '   or 'critical '
    :param format_str: The format to use. If not specified it will default to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    :return:
    """
    if format_str is None:
        format_str = '%(asctime)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format=format_str,
        filemode='a'
    )

configure_logging()

def __retry__(max_attempts=3, delay=1):
    """
    Decorator to retry a function if it fails within max_attempts times.
    :param max_attempts: Maximum number of attempts to retry the function.
    :param delay: Delay between attempts in seconds.
    :return: Decorated function with retry logic.
    """
    def process(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts}/{max_attempts} failed with error: {e}")
                    time.sleep(delay)
            print(f"Failed to execute function after {max_attempts} attempts")
        return wrapper
    return process


def __cache__(maxsize=128):
    """
    Decorator to wrap a function with an LRU cache.
    :param maxsize: the maximum size of the cache (default 128)
    :return: a decorated function that caches the return value of the function call after the first call has been made on the function
    """
    cache = {}
    keys_queue = []

    def process(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))

            if key in cache:
                # Move the key to the end of the queue (most recently used)
                keys_queue.remove(key)
                keys_queue.append(key)
                return cache[key]
            else:
                result = func(*args, **kwargs)
                cache[key] = result
                keys_queue.append(key)

                # If the cache size exceeds maxsize, remove the least recently used item
                if len(keys_queue) > maxsize:
                    old_key = keys_queue.pop(0)
                    del cache[old_key]

                return result

        return wrapper

    return process


def __log__(func):
    """
    Decorator to log the execution time of a function.  The log file is written to the current working directory.
    :param func: The function to be logging
    :return: The decorated function to be logged to the log file
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logging.info(f"Function {func.__name__} called with args: {args}, kwargs: {kwargs}, result: {result}")
        return result

    return wrapper

def __valid_type__(*arg_types, **kwarg_types):
    """
    Validates the input arguments for a function. If the input arguments are invalid, it raises an exception and returns None.
    :param arg_types: the input arguments types to validate against the input arguments types list in arg_types
    :param kwarg_types: the input arguments types to validate against the input arguments types list in kwarg_types (optional)
    :return: the validated input arguments or None if the input arguments are invalid
    """
    def process(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg, arg_type in zip(args, arg_types):
                if not isinstance(arg, arg_type):
                    raise TypeError(f"Expected {arg_type} for argument {arg}, got {type(arg)}")
            for kwarg, kwarg_type in kwarg_types.items():
                if kwarg in kwargs and not isinstance(kwargs[kwarg], kwarg_type):
                    raise TypeError(f"Expected {kwarg_type} for keyword argument {kwarg}, got {type(kwargs[kwarg])}")
            return func(*args, **kwargs)

        return wrapper

    return process

def __single__(cls):
    """
    Decorator to make a class a singleton. The class must be a subclass of object.
    :param cls: The class to make a singleton. Must be a subclass of object (not a subclass of object)
    :return: The decorated class that is a singleton class (not a subclass of object)
    """

    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = func(*args, **kwargs)
        return instances[cls]

    return wrapper

@__timer__
def load_datasets(_workspace, _project, _version, _type_model):
    """
    Load datasets from workspace and project into memory
    :param _workspace: workspace roboflow
    :param _project: project roboflow
    :param _version: version project
    :param _type_model: model type of datasets
    :return: list of datasets loaded from workspace and project into memory as dict {dataset_name: dataset}
    """
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(_workspace).project(_project)
    version = project.version(_version)
    dataset = version.download(_type_model)


@__timer__
def check_dataset(dataset: str) -> None:
    """
    Check if dataset is valid and exists inside datasets folder of workspace and project
    :param dataset: dataset name to check if exists inside datasets folder of workspace and project
    :return: None
    """
    if not os.path.isfile(dataset):
        with open(dataset, 'w') as file:
            file.write("./bus.jpg\n")

        print(f"File '{dataset}' did not exist and was created.")
    else:
        print(f"File '{dataset}' already exists.")

@__timer__
def get_filename_without_extension(filepath):
    """
    Get filename without extension from filepath and return it as string in correct format according to OS (Windows, Linux, Mac)
    :param filepath: filepath to get filename without extension from it and return it as string in correct format according to OS (Windows, Linux, Mac)
    :return: filename in correct format according to OS (Windows , Linux, Mac)
    """
    filename = os.path.basename(filepath)
    filename_without_extension = os.path.splitext(filename)[0]
    return filename_without_extension

@__timer__
def checking_dir(name_dirs):
    """
    Check if directory exists and if not create it if it doesn't exist and create it if it doesn't exist and return it as string in correct format according to OS (Windows, Linux, Mac)
    :param name_dirs: directory names to check if exists and return it as string in correct format according to OS (Windows, Linux, Mac)
    :return: directory name in correct format according to OS (Windows, Linux, Mac) or None if directory doesn't exist
    """
    for directory in name_dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' did not exist and was created.")
        else:
            print(f"Directory '{directory}' already exists.")

