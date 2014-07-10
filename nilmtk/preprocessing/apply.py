from __future__ import print_function, division
from ..node import Node

class Apply(Node):
    
    """Apply an arbitrary function to each pd.Series chunk."""

    def __init__(self, upstream=None, generator=None, func=None):
        self.func = func
        super(Apply, self).__init__(upstream, generator)

    def process(self):
        self.check_requirements()
        for chunk in self.upstream.process():
            yield self.func(chunk)
