from warnings import warn
from collections import namedtuple
from copy import deepcopy
from .hashable import Hashable
from .utils import flatten_2d_list
from nilm_metadata import get_appliance_types

ApplianceID = namedtuple('ApplianceID', ['type', 'instance'])
DEFAULT_ON_POWER_THRESHOLD = 10

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

    #: Static variable.  Controls whether Appliance.matches() allows synonyms
    #: for appliance type names.
    allow_synonyms = True

    def __init__(self, metadata=None):
        self.metadata = {} if metadata is None else metadata

        # Instantiate static appliance_types
        if not Appliance.appliance_types:
            Appliance.appliance_types = get_appliance_types()

        # Check appliance type
        if (self.identifier.type and
                self.identifier.type not in Appliance.appliance_types):
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
        """Return number of meters (int) to which this
        appliance is connected"""
        return len(self.metadata['meters'])

    def on_power_threshold(self):
        try:
            return self.metadata['on_power_threshold']
        except KeyError:
            pass

        try:
            return self.metadata['nominal_consumption']['on_power']
        except KeyError:
            threshold_from_appliance_type = self.type.get(
                'on_power_threshold', DEFAULT_ON_POWER_THRESHOLD)
            return threshold_from_appliance_type

    def label(self, pretty=False):
        """Return string '(<type>, <identifier>)' e.g. '(fridge, 1)'
        if `pretty=False` else if `pretty=True` then return a string like
        'Fridge' or 'Fridge 2'. If type == 'unknown' then 
        appends `original_name` to end of label."""
        if pretty:
            label = str(self.identifier.type)
            label = label.capitalize()
            if self.identifier.instance > 1:
                label += " {}".format(self.identifier.instance)
        else:
            label = str(tuple(self.identifier))

        if self.identifier.type == 'unknown':
            label += ', original name = {}'.format(
                self.metadata.get('original_name'))
        return label

    def categories(self):
        """Return 1D list of category names (strings)."""
        return flatten_2d_list(self.type.get('categories').values())

    def matches(self, key):
        """Returns True if all key:value pairs in `key` match 
        `appliance.metadata` or
        `Appliance.appliance_types[appliance.metadata['type']]`.
        Returns True if key is empty dict.

        By default, matches synonyms.  Set `Appliance.allow_synonyms = False`
        if you do not want to allow synonyms.

        Parameters
        ----------
        key : dict

        Returns
        -------
        Bool
        """
        if not key:
            return True

        if not isinstance(key, dict):
            raise TypeError()

        match = True
        for k, v in key.items():
            if hasattr(self.identifier, k):
                if Appliance.allow_synonyms and k == 'type':
                    synonyms = self.type.get('synonyms', [])
                    synonyms.append(self.identifier.type)
                    if v not in synonyms:
                        match = False
                elif getattr(self.identifier, k) != v:
                    match = False

            elif k in self.metadata:
                if self.metadata[k] != v:
                    match = False

            elif k == 'category':
                if v not in self.categories():
                    match = False

            elif k in self.type:
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
