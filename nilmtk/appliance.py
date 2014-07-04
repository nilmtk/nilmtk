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
        return _flatten(self.type.get('categories').values())

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
            if hasattr(self.identifier, k):
                if getattr(self.identifier, k) != v:
                    return False

            elif self.metadata.has_key(k):
                if self.metadata[k] != v:
                    return False

            elif k == 'category':
                if v not in self.categories():
                    return False

            elif self.type.has_key(k):
                metadata_value = self.type[k]
                if (isinstance(metadata_value, list) and 
                    not isinstance(v, list)):
                    # for example, 'control' is a list in metadata
                    if v not in metadata_value:
                        return False
                elif metadata_value != v:
                    return False

            else:
                return False

        return True
    

def _flatten(list2d):
    list1d = []
    for item in list2d:
        if isinstance(item, list):
            list1d.extend(item)
        else:
            list1d.append(item)
    return list1d
