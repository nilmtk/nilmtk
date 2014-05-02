from __future__ import print_function, division
import pandas as pd
from datetime import timedelta
from copy import deepcopy

class TimeFrame(object):
    """A TimeFrame is a single time span or period,
    e.g. from "2013" to "2014".

    Attributes
    ----------
    _start : pd.Timestamp or None
        if None and empty if False
        then behave as if start is infinitely far into the past
    _end : pd.Timestamp or None
        if None and empty is False 
        then behave as if end is infinitely far into the future
    enabled : boolean
        If False then behave as if both _end and _start are None
    _empty : boolean
        If True then represents an empty time frame
    """

    def __init__(self, start=None, end=None):
        self.clear()
        self.start = start
        self.end = end

    @property
    def start(self):
        if self.enabled:
            return self._start

    @property
    def end(self):
        if self.enabled:
            return self._end

    @property
    def empty(self):
        return self._empty 
          
    @start.setter
    def start(self, new_start):
        if new_start is None:
            self._start = None
            return
        new_start = pd.Timestamp(new_start)
        if self.end and new_start >= self.end:
            raise ValueError("start date must be before end date")
        else:
            self._start = new_start

    @end.setter
    def end(self, new_end):
        if new_end is None:
            self._end = None
            return
        new_end = pd.Timestamp(new_end)
        if self.start and new_end <= self.start:
            raise ValueError("end date must be after start date")
        else:
            self._end = new_end

    def adjacent(self, other, gap=0):
        """Returns True if self.start == other.end or visa versa.

        Parameters
        ----------
        gap : float or int
            Number of seconds gap allowed.
        """        
        assert gap >= 0
        gap_td = timedelta(seconds=gap)
        return (other.start - gap_td <= self.end <= other.start or
                self.start - gap_td <= other.end <= self.start)

    def union(self, other):
        """Return a single TimeFrame combining self and other."""
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        return TimeFrame(start, end)

    @property
    def timedelta(self):
        if self.end and self.start:
            return self.end - self.start

    def intersect(self, other):
        """Returns a new TimeFrame of the intersection between
        this TimeFrame and `other` TimeFrame.
        If the intersect is empty then the returned TimeFrame
        will have empty == True."""
        if other is None:
            return deepcopy(self)

        assert isinstance(other, TimeFrame)

        if self.empty or other.empty:
            start = None
            end = None
            empty = True
        else:
            if other.start is None:
                start = self.start
            elif self.start is None:
                start = other.start
            else:
                start = max(self.start, other.start)

            if other.end is None:
                end = self.end
            elif self.end is None:
                end = other.end
            else:
                end = min(self.end, other.end)

            empty = False

            if (start is not None) and (end is not None):
                if start >= end:
                    start = None
                    end = None
                    empty = True
        
        intersect = TimeFrame(start, end)
        intersect._empty = empty
        return intersect

    def query_terms(self, variable_name='timeframe'):
        if self.empty:
            raise Exception("TimeFrame is empty.")
        terms = []
        if self.start is not None:
            terms.append("index>=" + variable_name + ".start")
        if self.end is not None:
            terms.append("index<" + variable_name + ".end")
        return terms

    def clear(self):
        self.enabled = True
        self._start = None
        self._end = None
        self._empty = False

    def __nonzero__(self):
        if self.empty:
            return False
        else:
            return (self.start is not None) or (self.end is not None)

    def __repr__(self):
        return ("TimeFrame(start={}, end={}, empty={})"
                .format(self.start, self.end, self.empty))

    def __eq__(self, other):
        return ((other.start == self.start) and 
                (other.end == self.end) and
                (other.empty == self.empty))
