import os

def get_immediate_subdirectories(dir):
    # From Richie Hindle's StackOverflow answer: 
    # http://stackoverflow.com/a/800201/732596
    if dir:
        subdirs = [name for name in os.listdir(dir)
                   if os.path.isdir(os.path.join(dir, name))]
    else:
        subdirs = []
    return subdirs

