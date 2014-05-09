class Appliance(object):
    """
    Attributes
    ----------
    
    metadata : pd.DataFrame single row and columns:
       type : string (e.g. 'fridge' or 'television')
       instance : int (starting at 1)
       dataset : string (e.g. 'REDD')
       building : int (starting at 1)
       
       Only need to specify name & instance.  Then NILMTK will get the generic metadata
       for that name from the central appliance database.  Any additional
       metadata will override defaults.  e.g.:
       
       on_power_threshold : float, watts
       minimum_off_duration : timedelta
       minimum_on_duration : timedelta       
    """
    
    # TODO: appliance_types will be loaded from disk
    # just hard coding for now to get MVP finished.
    appliance_types = {'fridge': {'category': 'cold'}}

    def __init__(self, **metadata):
        if not all([metadata.has_key(k) for k in ['type', 'instance']]):
            raise ValueError("An Appliance must have a 'type' and 'instance'.")
        if not Appliance.appliance_types.has_key(metadata['type']):
            raise ValueError("'{}' is not a recognised appliance type."
                             .format(metadata['type']))
        self.metadata = metadata

    @property
    def type(self):
        return Appliance.appliance_types[self.metadata['type']]

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
            if self.metadata.has_key(k):
                if self.metadata[k] != v:
                    return False
            else:
                if self.type.get(k) != v:
                    return False
        return True

    @property
    def id(self):
        key_attrs = ['type', 'instance', 'dataset', 'building']
        id_dict = {}
        for key in key_attrs:
            id_dict[key] = self.metadata.get(key)
        return id_dict

    def __eq__(self, other):
        if isinstance(other, Appliance):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return hash(((k,v) for k,v in self.id.iteritems()))

    def __repr__(self):
        md = self.metadata
        return ("{:s}(type={}, instance={}, dataset={}, building={})"
                .format(self.__class__.__name__, 
                        md.get('type'), md.get('instance'), 
                        md.get('dataset'), md.get('building')))

    
