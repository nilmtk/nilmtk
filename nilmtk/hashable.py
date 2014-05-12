from collections import namedtuple

class Hashable(object):
    """Simple mix-in class to add functions necessary to make
    an object hashable.  Just requires the child class to have
    a `KEY_ATTRIBUTES` static variable and a `metadata` dict."""

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.identifier == other.identifier
        else:
            return False

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        s = "{:s}(".format(self.__class__.__name__)
        s += str(self.identifier).partition("(")[2]
        return s
