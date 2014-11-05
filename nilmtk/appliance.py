from __future__ import print_function, division
from warnings import warn
from collections import namedtuple
from copy import deepcopy
from .hashable import Hashable
from .utils import flatten_2d_list
from nilm_metadata import get_appliance_types

ApplianceID = namedtuple('ApplianceID', ['type', 'instance'])

class Appliance(Hashable):
    """Represents an appliance instance.

    Attributes
    ----------
    metadata : dict
       See here metadata attributes:
       http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#appliance
    """

    #: Static (AKA class) variable. Maps from appliance_type (string) to a dict
    #: describing metadata about each appliance type.
    appliance_types = {} 

    def __init__(self, metadata=None):
        self.metadata = {} if metadata is None else metadata

        # Instantiate static appliance_types
        if not Appliance.appliance_types:
            Appliance.appliance_types = get_appliance_types()

        # Check appliance type
        if (self.identifier.type and 
            not Appliance.appliance_types.has_key(self.identifier.type)):
            warn("'{}' is not a recognised appliance type."
                 .format(self.identifier.type), RuntimeWarning)

    @property
    def identifier(self):
        """Return ApplianceID"""
        md = self.metadata
        return ApplianceID(md.get('type'), md.get('instance'))

    @property
    def type(self):
        """Return deepcopy of dict describing appliance type."""
        return deepcopy(Appliance.appliance_types[self.identifier.type])

    @property
    def n_meters(self):
        """Return number of meters (int) to which this appliance is connected"""
        return len(self.metadata['meters'])

    def label(self):
        """Return string '(<type>, <identifier>)' e.g. '(fridge, 1)'.
        If type == 'unknown' then also returns `original_name`."""
        label = str(tuple(self.identifier))
        if self.identifier.type is 'unknown':
            label += ', original name = {}'.format(
                self.metadata.get('original_name'))
        return label

    def categories(self):
        """Return 1D list of category names (strings)."""
        return flatten_2d_list(self.type.get('categories').values())

    def matches(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        Bool
            True if all key:value pairs in `key` match `appliance.metadata`
            or `Appliance.appliance_types[appliance.metadata['type']]`.
            Returns True if key is empty dict.
        """
        if not key:
            return True

        if not isinstance(key, dict):
            raise TypeError()

        match = True
        for k, v in key.iteritems():
            if hasattr(self.identifier, k):
                if getattr(self.identifier, k) != v:
                    match = False

            elif self.metadata.has_key(k):
                if self.metadata[k] != v:
                    match = False

            elif k == 'category':
                if v not in self.categories():
                    match = False

            elif self.type.has_key(k):
                metadata_value = self.type[k]
                if (isinstance(metadata_value, list) and 
                    not isinstance(v, list)):
                    # for example, 'control' is a list in metadata
                    if v not in metadata_value:
                        match = False
                elif metadata_value != v:
                    match = False

            else:
                raise KeyError("'{}' not a valid key.".format(k))

        return match
