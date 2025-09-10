
# Record elapsed time of critical operations, note not thread safe

from time import time
import threading
import queue
from typing import Callable, Any, Optional


class MetaTiming(type):
    """
        Overwrites the __repr__() method so that print(Timing) works as
        required.
    """
    def __repr__(cls):
        return cls.to_string()


class Timing(metaclass=MetaTiming):
    """
        Singleton class collects count, mean and max time for actions.

        :cvar bool active: whether timing collection is active
        :cvar times dict: times collected
                          {action1: {scale1: {count, total, max}, ...}, ...}
        :cvar set/None filter: only these actions will be recorded
    """
    MAX_ACTION_LEN = 10
    VAL_FMT = '{}  {}    {:11.0f}    {:12.3f}    {:12.3f}    {:12.3f}  \n'

    active = False
    times = {}
    filter = None

    @classmethod
    def on(self, active, filter=None):
        """
            Switching timing on and off

            :param bool active: whether timing should be on or off
            :param set/None filter: only these actions recorded

            :raises TypeError: if bad arg type
        """
        if (not isinstance(active, bool) or
            (filter is not None and
             (not isinstance(filter, set)
              or not all([isinstance(f, str) for f in filter])))):
            raise TypeError('Timing.on() bad arg type')
        self.active = active
        self.filter = filter
        self.times = {}

    @classmethod
    def now(self):
        """
            Returns current time, generally for the start of an action

            :returns float: epoch time in seconds
        """
        return time() if self.active is True else None

    @classmethod
    def record(self, action, scale, start):
        """
            Records time for specified action

            :param str action: action being timed
            :param int scale: indication of scale of action e.g. num of nodes
            :param int start: time at which action started

            :raises TypeError: if bad arg type
            :raises ValueError: if bad arg value

            :returns float: epoch time when this function called
        """
        now = None
        if (self.active is True
                and (self.filter is None or action in self.filter)):
            if action not in self.times:
                if not isinstance(action, str):
                    raise TypeError('Timing.record() bad arg type')
                if len(action) == 0 or len(action) > self.MAX_ACTION_LEN:
                    raise ValueError('Timing.record() bad arg value')
                self.times[action] = {}

            if scale not in self.times[action]:
                if not isinstance(scale, int) or isinstance(scale, bool):
                    raise TypeError('Timing.record() bad arg type')
                self.times[action].update({scale: {'count': 0, 'total': 0.0,
                                                   'max': 0.0}})

            if not isinstance(start, float):
                raise TypeError('Timing.record() bad arg type')

            now = time()
            elapsed = now - start
            self.times[action][scale]['count'] += 1
            self.times[action][scale]['total'] += elapsed
            if elapsed > self.times[action][scale]['max']:
                self.times[action][scale]['max'] = elapsed

        return now

    @classmethod
    def to_string(cls, filter=None):
        """
            Print out timings in nice format, optionally only for specified
            actions.

            :param set/None filter: only return info about these actions.

            :return str: human friendly timing information
        """
        if cls.active is True:
            if (filter is not None and
                    (not isinstance(filter, set)
                     or not all([isinstance(f, str) for f in filter]))):
                raise TypeError('Timing.to_string() bad arg value')

            # Column headers

            res = ('\n{:^{}s}{:^14}{:^15}{:^16}{:^16}{:^16}\n'
                   .format('Action', cls.MAX_ACTION_LEN + 4, 'Scale',
                           'Count', 'Mean (s)', 'Max. (s)', 'Total (s)'))
            res += ((' {0:->{1}s}{2}{0:->12}{2}{0:->13}{2}{0:->14}' +
                     '{2}{0:->14}{2}{0:->14}\n')
                    .format('-', cls.MAX_ACTION_LEN + 2, '  '))

            for action in sorted(Timing.times):
                if filter is not None and action not in filter:
                    continue

                a_str = '  {:>{}s}  '.format(action, cls.MAX_ACTION_LEN)
                count = 0
                num = 0
                total = 0.0
                max = None
                for scale in sorted(Timing.times[action]):
                    v = Timing.times[action][scale]
                    num += 1
                    count += v['count']
                    total += v['total']
                    max = v['max'] if max is None or v['max'] > max else max
                    res += (cls.VAL_FMT
                            .format(a_str, '{:10.0f}'.format(scale),
                                    v['count'], v['total'] / v['count'],
                                    v['max'], v['total']))
                if num > 1:
                    res += cls.VAL_FMT.format(a_str, '       ALL', count,
                                              total / count, max, total)
                res += '\n'

        else:
            res = '\n\nTiming was not enabled.\n'

        return res

    @classmethod
    def __repr__(cls):
        """
            Print out all timings in nice format

            :return str: human friendly timing information
        """
        return cls.to_string()


# Timeout functionality for algorithm execution

class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass


def run_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None,
                     timeout_seconds: Optional[int] = None) -> Any:
    """
    Run a function with a timeout using threading approach.

    This is compatible with both subprocess-based (R, Java) and direct Python
    function calls (causal-learn). For subprocess calls, the timeout will
    terminate the subprocess if it exceeds the time limit.

    :param func: Function to execute
    :param args: Positional arguments for the function
    :param kwargs: Keyword arguments for the function
    :param timeout_seconds: Maximum execution time in seconds,
                           None for no timeout

    :raises TimeoutError: if execution exceeds timeout_seconds
    :raises Exception: any exception raised by the wrapped function

    :returns: Result from the wrapped function
    """
    if timeout_seconds is None:
        # No timeout specified, run directly
        return func(*(args or ()), **(kwargs or {}))

    if timeout_seconds <= 0:
        raise ValueError("Timeout must be positive")

    result_queue = queue.Queue()
    exception_queue = queue.Queue()

    def target():
        """Target function that runs in separate thread"""
        try:
            result = func(*(args or ()), **(kwargs or {}))
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)

    # Start the function in a separate thread
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()

    # Wait for completion or timeout
    thread.join(timeout_seconds)

    if thread.is_alive():
        # Thread is still running - timeout occurred
        # Note: We can't actually kill the thread, but we can abandon it
        # For subprocess-based algorithms, this will leave a subprocess running
        # That's a limitation we'll need to address in subprocess wrappers
        raise TimeoutError(f"Algorithm execution exceeded "
                           f"{timeout_seconds} seconds")

    # Check if an exception occurred
    if not exception_queue.empty():
        raise exception_queue.get()

    # Check if we have a result
    if not result_queue.empty():
        return result_queue.get()

    # This shouldn't happen, but just in case
    raise RuntimeError("Thread completed but no result or exception found")


def with_timeout(timeout_seconds: Optional[int] = None):
    """
    Decorator version of run_with_timeout.

    :param timeout_seconds: Maximum execution time in seconds,
                           None for no timeout

    Usage:
        @with_timeout(300)  # 5 minute timeout
        def my_algorithm(data):
            # algorithm implementation
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return run_with_timeout(func, args, kwargs, timeout_seconds)
        return wrapper
    return decorator
