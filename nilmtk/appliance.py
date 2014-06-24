from __future__ import print_function, division
from warnings import warn
from .hashable import Hashable
from collections import namedtuple

ApplianceID = namedtuple('ApplianceID', ['type', 'instance'])

class Appliance(Hashable):
    """
    Attributes
    ----------
    metadata : dict
       see here metadata attributes:
       http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#appliance
    """

    # TODO: appliance_types will be loaded from disk
    # just hard coding for now to get MVP finished.
    appliance_types = {'fridge': {'category': 'cold'}}

    def __init__(self, metadata=None):
        self.metadata = {} if metadata is None else metadata
        if (self.identifier.type and 
            not Appliance.appliance_types.has_key(self.identifier.type)):
            warn("'{}' is not a recognised appliance type."
                 .format(self.identifier.type), RuntimeWarning)

    @property
    def identifier(self):
        md = self.metadata
        return ApplianceID(md.get('type'), md.get('instance'))

    @property
    def type(self):
        return Appliance.appliance_types[self.identifier.type]

    @property
    def n_meters(self):
        return len(self.metadata['meters'])

    def label(self):
        return str(tuple(self.identifier))

    def matches(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        True if all key:value pairs in `key` match `appliance.metadata`
        or `Appliance.appliance_types[appliance.metadata['type']]`.
        """
        if not isinstance(key, dict):
            raise TypeError()
        for k, v in key.iteritems():
            try:
                if getattr(self.identifier, k) != v:
                    return False
            except AttributeError:
                if self.metadata.has_key(k):
                    if self.metadata[k] != v:
                        return False
                else:
                    if self.type.get(k) != v:
                        return False
        return True
    
