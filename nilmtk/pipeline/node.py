import abc

class UnsatisfiedRequirementsError(Exception):
    pass

def find_unsatisfied_requirements(state, requirements):
    """
    Parameters
    ----------
    state, requirements : dict
        If a property is required but the specific value does not
        matter then use 'ANY VALUE' as the value in `requirements`.

    Returns
    -------
    list of strings describing (for human consumption) which 
    conditions are not satisfied.  If all conditions are satisfied
    then returns an empty list.
    """
    unsatisfied = []
    def unsatisfied_requirements(cond, req):
        # Recursively find requirements
        for key, value in req.iteritems():
            try:
                cond_value = cond[key]
            except KeyError:
                msg = ("Requires '{}={}' but '{}' not in state dict."
                       .format(key, value, key))
                unsatisfied.append(msg)
            else:
                if isinstance(value, dict):
                    unsatisfied_requirements(cond_value, value)
                elif value != 'ANY VALUE' and cond_value != value:
                    msg = ("Requires '{}={}' not '{}={}'."
                           .format(key, value, key, cond_value))
                    unsatisfied.append(msg)

    unsatisfied_requirements(state, requirements)

    return unsatisfied
    

class Node(object):
    """Abstract class defining interface for all Node subclasses,
    where a 'node' is a module which runs pre-processing or statistics
    (or, later, maybe NILM training or disaggregation).
    """

    __metaclass__ = abc.ABCMeta

    requirements = {}
    postconditions = {}

    def __init__(self, name):
        self.name = name

    def update_state(self, state):
        """Recursively updates `state` dict with `postconditions`.

        This function is required because Python's `dict.update()` function
        does not descend into dicts within dicts.

        Parameters
        ----------
        state : dict

        Returns
        -------
        state : dict
        """
        def _update_state(state, postconditions):
            # Recursively update dict.
            for key, value in postconditions.iteritems():
                try:
                    state_value = state[key]
                except KeyError:
                    state[key] = value
                else:
                    if isinstance(value, dict):
                        _update_state(state_value, value)
                    else:
                        state[key] = value
            return state
        return _update_state(state, self.postconditions)

    def check_requirements(self, state):
        """
        Parameters
        ----------
        state : dict
        
        Raises
        ------
        UnsatistfiedPreconditionsError
        
        Description
        -----------
        
        Requirements can be of the form:
    
        "node X needs (power.apparent or power.active) (but not
        power.reactive) and voltage is useful but not essential"
    
        or
    
        "node Y needs everything available from disk (to save to a copy to
        disk)"
    
        or
    
        "ComputeEnergy node needs good sections to be located" (if
        none of the previous nodes provide this service then check
        source.metadata to see if zeros have already been inserted; if the
        haven't then raise an error to tell the user to add a
        LocateGoodSectionsNode.)
        """
        # If a subclass has complex rules for preconditions then
        # override this default method definition.
        unsatisfied = find_unsatisfied_requirements(state, self.requirements)
        if unsatisfied:
            msg = str(self) + " not satisfied by:\n" + str(unsatisfied)
            raise UnsatisfiedRequirementsError(msg)
            
    def required_measurements(self, state):
        """
        Returns
        -------
        Set of measurements that need to be loaded from disk for this node.
        """
        return set()

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.name

    @abc.abstractmethod
    def process(self, df, metadata):
        # check_preconditions again??? (in case this node is not run in
        # the context of a Pipeline?)
        # do stuff to df
        return df
