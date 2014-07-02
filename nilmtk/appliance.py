from __future__ import print_function, division
from warnings import warn
from .hashable import Hashable
from collections import namedtuple
from nilm_metadata import get_appliance_types

ApplianceID = namedtuple('ApplianceID', ['type', 'instance'])

class Appliance(Hashable):
    """
    Attributes
    ----------
    metadata : dict
       see here metadata attributes:
       http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#appliance
    """

    appliance_types = get_appliance_types()

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

    def categories(self):
        return _flattern(self.type.get('categories').values())

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
                    if k == 'category':
                        if v not in self.categories():
                            return False
                    elif self.type.get(k) != v:
                        return False
        return True
    

def _flattern(list2d):
    list1d = []
    for item in list2d:
        if isinstance(item, list):
            list1d.extend(item)
        else:
            list1d.append(item)
    return list1d
