import time
from functools import lru_cache

__all__ = ['profile', 'cached']

def profile(fun):
    name = repr(fun.__name__)
    def wrap(*args, **kwargs):
        print(f'Running {name}...')
        t1 = time.time()
        result = fun(*args, **kwargs) 
        t2 = time.time()
        print(f'  {name} executed in {(t2-t1):.4f}s')
        return result
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
    elif hasattr(a, '__len__'):
        return tuple(deep_list_to_tuple(b) for b in a)
    else:
        raise TypeError(f'Cannot convert value to a hashable type (value = {a})')


def convert_list_arguments_to_tuples(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i in range(len(args)):
            args[i] = deep_list_to_tuple(args[i])
        for key,value in kwargs.items():
            kwargs[key] = deep_list_to_tuple(value)
        return func(*args, **kwargs)
    return wrapper


def cached(fun):
    return convert_list_arguments_to_tuples(lru_cache(maxsize=None)(fun))

