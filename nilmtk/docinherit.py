"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

class Baz(Bar):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
and  Baz.foo.__doc__ == Baz().foo.__doc__ == Bar.foo.__doc__ == Foo.foo.__doc__

From: http://code.activestate.com/recipes/576862-docstring-inheritance-decorator/
and: https://stackoverflow.com/a/38414303/8289769
"""
import inspect


from functools import wraps


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            if obj:
                return self.mthd(obj, *args, **kwargs)
            else:
                return self.mthd(*args, **kwargs)

        doc = inspect.getdoc(overridden)
        f.__doc__ = doc
        return f


doc_inherit = DocInherit

