# python
from typing import get_type_hints
import inspect

# python
from typing import Any, Callable, Dict, Tuple, get_type_hints


# python
class Inject:
    """
    Sentinel used as a default for attributes that should be injected.
    Example:
      service: IService = Inject()
    """
    def __repr__(self):
        return "<Inject>"


class Container:
    """
    Simple DI container: register(interface, provider, singleton=False)
    provider may be a class or a zero-arg factory (callable returning instance).
    """
    def __init__(self):
        self._providers: Dict[Any, Tuple[Callable[..., Any], bool]] = {}
        self._singletons: Dict[Any, Any] = {}

    def register(self, interface: Any, provider: Callable[..., Any] = None, *, singleton: bool = False):
        if provider is None:
            provider = interface
        self._providers[interface] = (provider, singleton)

    def resolve(self, interface: Any):
        # return existing singleton
        if interface in self._singletons:
            return self._singletons[interface]
        entry = self._providers.get(interface)
        if entry is None:
            raise KeyError(f"No provider registered for {interface}")
        provider, singleton = entry
        instance = provider() if callable(provider) else provider
        if singleton:
            self._singletons[interface] = instance
        return instance

    def inject_into(self, instance: Any):
        """
        Attribute injection: for attributes annotated on the class, if current
        value is an Inject sentinel or None, resolve and set.
        """
        hints = get_type_hints(instance.__class__, include_extras=False)
        for name, typ in hints.items():
            cur = getattr(instance, name, None)
            # check sentinel by exact type name or None
            if isinstance(cur, Inject) or cur is None:
                try:
                    val = self.resolve(typ)
                except KeyError:
                    continue
                setattr(instance, name, val)


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
