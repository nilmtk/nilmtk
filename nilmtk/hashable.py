class Hashable(object):
    """Simple mix-in class to add functions necessary to make
    an object hashable.  Just requires the child class to have
    an `identifier` namedtuple."""

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.identifier == other.identifier
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        s = "{:s}(".format(self.__class__.__name__)
        s += str(self.identifier).partition("(")[2]
        return s
