# python
from typing import Any, Callable, Dict, Tuple, Type, get_type_hints
import inspect

from hikerverseuniverse.di_inject import Inject


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
