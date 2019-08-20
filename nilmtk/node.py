from copy import deepcopy
from nilm_metadata import recursively_update_dict

class Node(object):
    """Abstract class defining interface for all Node subclasses,
    where a 'node' is a module which runs pre-processing or statistics
    (or, later, maybe NILM training or disaggregation).
    """

    requirements = {}
    postconditions = {}
    results_class = None

    def __init__(self, upstream=None, generator=None):
        """
        Parameters
        ----------
        upstream : an ElecMeter or MeterGroup or a Node subclass
            Required methods:
            - dry_run_metadata
            - get_metadata
            - process (not required if `generator` supplied)
        generator : Python generator. Optional
            Used when `upstream` object is an ElecMeter or MeterGroup.
            Provides source of data.
        """
        self.upstream = upstream
        self.generator = generator
        self.results = None
        self.reset()

    def reset(self):
        if self.results_class is not None:
            self.results = self.results_class()

    def process(self):
        return self.generator # usually overridden by subclass

    def run(self):
        """Pulls data through the pipeline.  Useful if we just want to calculate 
        some stats."""
        for _ in self.process():
            pass

    def check_requirements(self):
        """Checks that `self.upstream.dry_run_metadata` satisfies `self.requirements`.

        Raises
        ------
        UnsatistfiedRequirementsError
        """
        # If a subclass has complex rules for preconditions then
        # override this method.
        unsatisfied = find_unsatisfied_requirements(self.upstream.dry_run_metadata(),
                                                    self.requirements)
        if unsatisfied:
            msg = str(self) + " not satisfied by:\n" + str(unsatisfied)
            raise UnsatisfiedRequirementsError(msg)
            
    def dry_run_metadata(self):
        """Does a 'dry run' so we can validate the full pipeline before
        loading any data.

        Returns
        -------
        dict : dry run metadata
        """
        state = deepcopy(self.__class__.postconditions)
        recursively_update_dict(state, self.upstream.dry_run_metadata())
        return state

    def get_metadata(self):
        if self.results:
            metadata = deepcopy(self.upstream.get_metadata())
            results_dict = self.results.to_dict()
            recursively_update_dict(metadata, results_dict)
        else:
            # Don't bother to deepcopy upstream's metadata if 
            # we aren't going to modify it.
            metadata = self.upstream.get_metadata()
        return metadata

    def required_measurements(self, state):
        """
        Returns
        -------
        Set of measurements that need to be loaded from disk for this node.
        """
        return set()


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

    def unsatisfied_requirements(st, req):
        # Recursively find requirements
        for key, value in req.items():
            try:
                cond_value = st[key]
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
