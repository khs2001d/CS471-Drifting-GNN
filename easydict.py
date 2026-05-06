# Minimal local fallback for the external easydict package used by DiffSTG.
class EasyDict(dict):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        mapping = mapping or {}
        self.update(mapping)
        self.update(kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, mapping=None, **kwargs):
        mapping = mapping or {}
        items = mapping.items() if hasattr(mapping, "items") else mapping
        for key, value in items:
            if isinstance(value, dict) and not isinstance(value, EasyDict):
                value = EasyDict(value)
            super().__setitem__(key, value)
        for key, value in kwargs.items():
            if isinstance(value, dict) and not isinstance(value, EasyDict):
                value = EasyDict(value)
            super().__setitem__(key, value)
        return None
