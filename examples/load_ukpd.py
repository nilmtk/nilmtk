from nilmtk.dataset.ukpd import UKPD
ukpd = UKPD()
ukpd.load('/data/mine/vadeec/merged', downsample_one_sec_mains_rule='6S')
ukpd.export('/data/mine/vadeec/h5')
print "done"
