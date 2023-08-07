import warnings
import functools


def get_pe_attribute(pe_name):
    """Returns the name of the attribute that stores the positional encoding
    for the given PE method."""
    if pe_name == 'rw':
        return 'random_walk_pe'
    elif pe_name == 'lap':
        return 'laplacian_eigenvector_pe'
    else:
        raise NotImplementedError(f"PE method \"{pe_name}\" not implemented.")

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func
