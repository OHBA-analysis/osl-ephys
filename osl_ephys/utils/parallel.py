"""Utility functions for parallel processing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>

from functools import partial
import dask.bag as db
import queue
from dask.distributed import Client, LocalCluster, Queue, wait, default_client
from distributed.worker import get_worker
import multiprocessing

# Housekeeping for logging
import logging
osl_logger = logging.getLogger(__name__)


def dask_parallel_bag(func, iter_args,
                      func_args=None, func_kwargs=None,
                      queue_name="MAIN_THREAD_TASKS"):
    """A maybe more consistent alternative to ``dask_parallel``.
    
    Parameters
    ---------
    func : function
        The function to run in parallel.
    iter_args : list
        A list of iterables to pass to func.
    func_args : list, optional
        A list of positional arguments to pass to func.
    func_kwargs : dict, optional
        A dictionary of keyword arguments to pass to func.
    
    Returns
    -------
    flags : list
        A list of return values from func.
        
    References
    ----------
    https://docs.dask.org/en/stable/bag.html
    
    """

    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs

    # Get connection to currently active cluster
    client = default_client()
    queue = Queue(name="MAIN_THREAD_TASKS")
    # # Initialize or retrieve the shared queue on the scheduler
    # try:
    #     queue = Queue(name=queue_name)
    #     osl_logger.info(f"Queue '{queue_name}' initialized.")
    # except ValueError:
    #     # If the queue doesn't exist, create it explicitly
    #     queue = Queue(name=queue_name, client=client)
    #     osl_logger.info(f"Queue '{queue_name}' created.")

    # Print some helpful info
    osl_logger.info('Dask Client : {0}'.format(client.__repr__()))
    osl_logger.info('Dask Client dashboard link: {0}'.format(client.dashboard_link))

    osl_logger.debug('Running function : {0}'.format(func.__repr__()))
    osl_logger.debug('User args : {0}'.format(func_args))
    osl_logger.debug('User kwargs : {0}'.format(func_kwargs))

    # Set kwargs - need to handle args on function call to preserve order.
    run_func = partial(func, **func_kwargs)
    osl_logger.info('Function defined : {0}'.format(run_func))

    # Ensure input iter_args is list of lists
    if all(isinstance(aa, (list, tuple)) for aa in iter_args) is False:
        iter_args = [[aa] for aa in iter_args]

    # Add fixed positonal args if specified
    if func_args is not None:
        iter_args = [list(aa) + func_args for aa in iter_args]

    # Make dask bag from inputs: https://docs.dask.org/en/stable/bag.html
    b = db.from_sequence(iter_args)

    # Map iterable arguments to function using dask bag + current client
    bm = b.starmap(run_func)

    # Actually run the computation
    flags = bm.compute()

    osl_logger.info('Computation complete')

    return flags

#%% Main thread scheduler
# Some tasks can't be run on individual workers, and have to be run on the main thread.

def is_dask_worker():
    """Check if the current code is running inside a Dask worker."""
    try:
        get_worker()  # Attempt to get the current worker
        return True
    except ValueError:
        return False  # Not running in a Dask worker

def schedule_or_execute_task(id, func, *args, **kwargs): 
    """Add a task to be executed on the main thread. If already on the main thread, execute it immediately."""
    if is_dask_worker():
        # Dynamically retrieve the shared queue
        queue = Queue('MAIN_THREAD_TASKS')
        task = (id, func, args, kwargs)  # Package task as tuple
        osl_logger.info(f"Delaying task <{func.__name__}> for subject '{id}' to be run on the main thread.")
        queue.put(task)  # Add task to queue
    else:
        func(*args, **kwargs)

def execute_main_thread_tasks(queue_name="MAIN_THREAD_TASKS"):
    """Execute all tasks stored in the shared queue."""
    queue = Queue('MAIN_THREAD_TASKS')  # Dynamically retrieve the shared queue
    while queue.qsize() > 0:
        id, func, args, kwargs = queue.get()  # Retrieve task from queue
        osl_logger.info(f"Executing task <{func.__name__}> for subject '{id}' on main thread")
        func(*args, **kwargs)  # Execute task on main thread