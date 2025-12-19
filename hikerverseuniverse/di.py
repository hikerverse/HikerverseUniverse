# python
from typing import Callable, get_type_hints
import inspect

from hikerverseuniverse.di_container import Container


def inject_constructor(container: Container):
    """
    Class decorator that wraps __init__ to auto-resolve annotated parameters
    not supplied by the caller (or supplied as None).
    Usage:
      @inject_constructor(container)
      class Consumer:
          def __init__(self, service: IService):
              ...
    """
    def decorator(cls):
        orig_init = cls.__init__
        sig = inspect.signature(orig_init)
        hints = get_type_hints(orig_init, include_extras=False)

        def __init__(self, *args, **kwargs):
            bound = sig.bind_partial(self, *args, **kwargs)
            # iterate parameters after 'self'
            for name, param in list(sig.parameters.items())[1:]:
                if name not in bound.arguments or bound.arguments.get(name) is None:
                    typ = hints.get(name)
                    if typ is None:
                        continue
                    try:
                        bound.arguments[name] = container.resolve(typ)
                    except KeyError:
                        # leave missing param as-is; original init may handle defaults
                        pass
            return orig_init(*bound.args, **bound.kwargs)

        cls.__init__ = __init__
        return cls
    return decorator
