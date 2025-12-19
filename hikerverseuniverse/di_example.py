# python
from typing import Protocol

from hikerverseuniverse.di import inject_constructor
from hikerverseuniverse.di_container import Container
from hikerverseuniverse.di_inject import Inject


class IService(Protocol):
    def do(self) -> str: ...


class ServiceImpl:
    def do(self) -> str:
        return "ok"


# create container and register
c = Container()
c.register(IService, ServiceImpl, singleton=True)


# constructor injection
@inject_constructor(c)
class ConsumerA:
    def __init__(self, svc: IService):
        self.svc = svc


a = ConsumerA()  # svc auto-resolved
print(a.svc.do())


# attribute injection
class ConsumerB:
    svc: IService = Inject()


b = ConsumerB()
c.inject_into(b)  # fills b.svc
print(b.svc.do())
