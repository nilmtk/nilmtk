from .hashable import Hashable
from collections import namedtuple

ApplianceID = namedtuple('ApplianceID', ['type', 'instance'])

class Appliance(Hashable):
    """
    Attributes
    ----------

    metadata : dict
       type : string (e.g. 'fridge' or 'television')
       instance : int (starting at 1)
       on_power_threshold : float, watts
       minimum_off_duration : timedelta
       minimum_on_duration : timedelta
    """

    # TODO: appliance_types will be loaded from disk
    # just hard coding for now to get MVP finished.
    appliance_types = {'fridge': {'category': 'cold'}}

    def __init__(self, type, instance, metadata=None):
        if not Appliance.appliance_types.has_key(type):
            raise ValueError("'{}' is not a recognised appliance type."
                             .format(type))
        self.metadata = {} if metadata is None else metadata
        self.metadata.update({'type': type, 'instance': instance})

    @property
    def identifier(self):
        md = self.metadata
        return ApplianceID(md.get('type'), md.get('instance'))

    @property
    def type(self):
        return Appliance.appliance_types[self.identifier.type]

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
    
