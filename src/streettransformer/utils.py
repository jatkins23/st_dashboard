import time
import functools
from typing import Callable, Any

def time_it(func: Callable) -> Callable:
    """Decorator that times search execution and adds metadata to results."""

    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any):
        start_time = time.perf_counter()
        results = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)

        return results, elapsed_time

    return wrapper