class Hashable(object):
    """Simple mix-in class to add functions necessary to make
    an object hashable.  Just requires the child class to have
    a `KEY_ATTRIBUTES` static variable and a `metadata` dict."""

    @property
    def id(self):
        id_dict = {}
        for key in self.__class__.KEY_ATTRIBUTES:
            id_dict[key] = self.metadata.get(key)
        return id_dict

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return hash((self.metadata.get(k) for k in self.__class__.KEY_ATTRIBUTES))

    def __repr__(self):
        s = "{:s}(".format(self.__class__.__name__)
        s += ", ".join(["{:s}={}".format(k, self.metadata.get(k))
                       for k in self.__class__.KEY_ATTRIBUTES])
        s += ")"
        return s
