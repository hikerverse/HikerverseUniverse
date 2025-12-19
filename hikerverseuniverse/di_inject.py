# python
class Inject:
    """
    Sentinel used as a default for attributes that should be injected.
    Example:
      service: IService = Inject()
    """
    def __repr__(self):
        return "<Inject>"
