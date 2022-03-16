import time
from functools import lru_cache

__all__ = ['profile', 'profile_method', 'cached']

def _profile(name, fun, *args, **kwargs):
    print(f'Running {name}...')
    t1 = time.time()
    result = fun(*args, **kwargs)
    t2 = time.time()
    duration = t2 - t1
    print(f'  {name} executed in {duration:.4f}s')
    return result


def profile(fun):
    name = repr(fun.__name__)
    def wrap(*args, **kwargs):
        return _profile(name, fun, *args, **kwargs)
    return wrap


def profile_method(fun):
    def wrap(self, *args, **kwargs):
        name = str(type(self)) + f'.{fun.__name__}'
        return _profile(name, fun, self, *args, **kwargs)
    return wrap


def hashable(a):
    try:
        hash(a)
        return True
    except TypeError:
        return False


def deep_list_to_tuple(a):
    if hashable(a):
        return a
    elif isinstance(a, dict):
        return tuple((deep_list_to_tuple(k), deep_list_to_tuple(v)) for k,v in a.items())
    elif hasattr(a, '__len__'):
        return tuple(deep_list_to_tuple(b) for b in a)
    else:
        raise TypeError(f'Cannot convert value to a hashable type (value = {a})')


def convert_list_arguments_to_tuples(fun):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            args[i] = deep_list_to_tuple(args[i])
        for key,value in kwargs.items():
            kwargs[key] = deep_list_to_tuple(value)
        return fun(*args, **kwargs)
    return wrapper


def cached(fun):
    return convert_list_arguments_to_tuples(lru_cache(maxsize=None)(fun))

