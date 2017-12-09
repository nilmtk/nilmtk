#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:06:59 2017

@author: martinneighbours

*** Objective ***

The purpose of this module is to create more readable api docs by
- including autosummaries at the top of the module before the detail of the module
- removing the modules section from intermediate levels with no modules

This is not appropriate for all modules so the min_members_to_document parameter can be
used to limit the modules that are modified.

Also not all modules will document with autosummary, so to avoid creating an error, an
exclude_list has been included which will default to modules

The core content of this module has been borrowed from sphinx-autosummary generate.py

Example
-------
>>> main(['modules','nilmtk.tests.test_datastore_converter'],5,'nilmtk')

"""

from __future__ import print_function

import os
import sys
import shlex, subprocess
import re

from jinja2 import FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
from sphinx.util.rst import escape as rst_escape

from sphinx.ext.autosummary import import_by_name, get_documenter
from sphinx.util.inspect import safe_getattr
# Add documenters to AutoDirective registry
from sphinx.ext.autodoc import add_documenter, \
    ModuleDocumenter, ClassDocumenter, ExceptionDocumenter, DataDocumenter, \
    FunctionDocumenter, MethodDocumenter, AttributeDocumenter, \
    InstanceAttributeDocumenter



add_documenter(ModuleDocumenter)
add_documenter(ClassDocumenter)
add_documenter(ExceptionDocumenter)
add_documenter(DataDocumenter)
add_documenter(FunctionDocumenter)
add_documenter(MethodDocumenter)
add_documenter(AttributeDocumenter)
add_documenter(InstanceAttributeDocumenter)

# Utility functions -----------------------------------------------------------

def get_class_that_defined_method(method):
    '''Return the fullname of the class that the method was created in
    
    Parameters
    ----------
    method: obj, method object
    
    Returns
    -------
    str
        The __name__ of the class that the method originated in 
    '''
    method_name = method.__name__

    try: 
        if method.__self__:    
            classes = [method.__self__.__class__]
        else:
            #unbound method
            classes = [method.im_class]
        while classes:
            c = classes.pop()
            if method_name in c.__dict__:
                return c.__name__
            else:
                classes = list(c.__bases__) + classes
    except:
        return 'Unable to determine class type'
    return None

# borrowed from sphinx-autosummary - amended quite a bit
def get_members(obj, typ, include_public=[], imported=False):
# type: (Any, unicode, List[unicode], bool) -> Tuple[List[unicode], List[unicode]]  # NOQA
   '''Document different members from a module
   
   Restricted to members from the same application as the object
   
   Parameters
   ----------
   obj: obj
       A module that can be imported
   typ: str
       {Function, class, exception, method, attribute}
   include_public: list, optional
       allows for sub-selection of public methods (default [])
   imported: bool, optional
       Whether to include imported members (default = False)
   
   Returns
   -------
   public: list
       List of public members for the chosen typ
   items: list
       List of all members for the chosen typ
   inherited_method_flag: bool
       True if any of the methods in the class are inherited

   Notes
   -----
   The original code has been borrowed from sphinx-autosummary generate.py and 
   amended quite a bit to try and accomodate methods and attributes (there was 
   a bug in the original code) which excluded these.
   
   Also amended to include a flag used to show if the class includes any 
   inherited methods from the same application.
   '''
   
   try:
        app_package = obj.__module__.split('.')[0]
   except:
        app_package = obj.__name__.split('.')[0]
   inherited_method_flag = False
   items = []  # type: List[unicode]
   for name in dir(obj):
        try:
            value = safe_getattr(obj, name)
            if typ != 'attribute' and (getattr(value, '__module__', None) 
                is None or getattr(value, '__module__', None).split('.')[0] 
                != app_package): #only consider members from same app
                continue
        except AttributeError:
            continue
        documenter = get_documenter(value, obj)
        if documenter.objtype == typ:
            if typ == 'method':
                orig_class = get_class_that_defined_method(value)
                if imported or orig_class == obj.__name__:
                    items.append(name)
                else: 
                    if orig_class is not None and \
                        orig_class != 'Unable to determine class type':
                        inherited_method_flag = True
            if typ == 'attribute':
                items.append(name)
            elif imported or getattr(value, '__module__', None) == obj.__name__:
                # skip imported members if expected
                items.append(name)
   public = [x for x in items
              if x in include_public or not x.startswith('_')]
   return public, items, inherited_method_flag


def process(cmd):
    '''Utility function to execute terminal commands and return the results
    
    Parameters
    ----------    
    cmd: str
        command to execture
    
    Returns
    -------
    out: str
        Returned stdout output
    err: str
        Returned stderr output
    errorcode: int
        Returned errorcode 0 = o.k.
    
    Example
    -------
    
    >>> cmd = "sphinx-apidoc -f -e -n -o source ../nilmtk"
    >>> stream, err, errcode = process(cmd)
    '''
    process = subprocess.Popen(cmd, shell=True,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)

    # wait for the process to terminate
    out, err = process.communicate()
    errcode = process.returncode
    return out, err, errcode

# iterate through each module etc and populate the members

def create_module_details(app_modules,exclude_list=['modules']):
    '''Scans through all modules and populates the member details
    
    Parameters
    ----------
    app_modules: list
        List of all modules in the app
    exclude_list: list
        List of modules which are incompatible with autosummary or summary
        tables are not required for. (defaul ['modules'])
    
    Returns
    -------
    List
        Lists of all functions, classes, exceptions and for each class sub
        lists of all the methods and attributes as well as a flag to show that
        the class contains inherited members
    '''
    app_module_details = []
    imported_members = False
    for app_module in app_modules:
        if app_module not in exclude_list:
            app_module_detail = [app_module,{'function':[],'class':[],'exception':[],'method':{},'Inherited method flag':{}, 'attribute':{},'Total members':0}]
            name, obj, parent, mod_name = import_by_name(app_module)
            for member in ['function','class','exception']:
                public, items, _ = get_members(obj, member , imported=imported_members)
                app_module_detail[1][member] = public
                app_module_detail[1]['Total members'] += len(public)
                # iterate through the classes
                for mod_class in app_module_detail[1]['class']:
                    for member in ['method','attribute']:
                        name, obj, parent, mod_name = import_by_name(app_module+'.'+mod_class)
                        public, items, inherited_method_flag = get_members(obj, member , imported=imported_members) 
                        app_module_detail[1][member][mod_class] = public
                        app_module_detail[1]['Total members'] += len(public)
                        if member == 'method':
                            app_module_detail[1]['Inherited method flag'][mod_class] = inherited_method_flag
            app_module_details.append(app_module_detail)
    return app_module_details

def insert_lines(member_list,member_type,padding=''):
    '''Utility function to deal with autosummaries in the rst file for different member types
    
    Uses a dictionary to map the source names to more user friendly display names.  This should
    be converted to a template.
    
    Parameters
    ----------
    member_list: list
        the members to be summarised
    member_type: str
        the type of member - function, class, exception, method, attribute
    padding: str
        used to ensure that the autosummary is in the same scope
    
    Returns
    -------
    str
        text to insert into the rst file
    '''
    member_desc = {'function':'Functions',
                   'class':'Classes',
                   'exception':'Exceptions',
                   'method':'Methods',
                   'attribute':'Attributes'}
    lines_to_insert = padding + '.. rubric:: ' + member_desc[member_type] + '\n' *2
    lines_to_insert += padding + '.. autosummary::' + '\n' * 2
    for member in member_list:
        lines_to_insert += padding + '    ' + member + '\n'
    lines_to_insert += '\n' * 2
    return lines_to_insert

def _underline(title, line='='):
    if '\n' in title:
        raise ValueError('Can only underline single lines')
    return title + '\n' + line * len(title)

def update_modules(app_module_details, min_members_to_document=5):
    '''  Inserts module and class updates to the module.rst files
    
    Makes use of module_class.tplt which is stored in source/_templates
    
    Parameters
    ----------
    app_module_details: list
        Contains all the functions, classes, exceptions and for each class a dictionary
        list of all the methods and attributes with a flag to show inherited methods
    min_members_to_document: int
        No changes will be applied to modules with less than this number of members
    '''
    # modify the files by finding the relevant automodule directive
    for app_module in app_module_details:
        if app_module[1]['Total members'] >= min_members_to_document:

            # setup template directory
            template_dirs = ['source/_templates']
            
            template_loader = FileSystemLoader(template_dirs)
            template_env = SandboxedEnvironment(loader=template_loader)
 
            template_env.filters['underline'] = _underline
                  
            # replace the builtin html filters
            template_env.filters['escape'] = rst_escape
            template_env.filters['e'] = rst_escape
            
            template_name = 'module_class.tplt'
            
            ns = {}  # type: Dict[unicode, Any]
            ns['functions'] = app_module[1]['function']
            ns['classes'] = app_module[1]['class']
            ns['exceptions'] = app_module[1]['exception']
            
            name = app_module[0]
            parts = name.split('.')
            mod_name, obj_name = '.'.join(parts[:-1]), parts[-1]
            
            ns['fullname'] = name
            ns['module'] = mod_name
            ns['objname'] = obj_name
            ns['name'] = parts[-1]
        
            ns['objtype'] = 'module'
            ns['underline'] = len(name) * '='
            ns['methods'] = app_module[1]['method']
            ns['attributes'] = app_module[1]['attribute']
            ns['inherited_method_flags'] = app_module[1]['Inherited method flag']
            
            template = template_env.get_template(template_name)
            rendered = template.render(**ns)

            filename = 'source/' + app_module[0] + '.rst' 
            print ('Modifying file :' + filename)
            with open(filename,'w') as f:
                f.write(rendered)
                
  
def remove_empty_module_contents(app_module_details):
    '''Removes the Module header from any rst files that don't have any modules
    
    Parameters
    ----------
    app_module_contents: list
        List of modules and their contents
    '''
    # remove module contents from any members that don't have modules
    # these will generally be sub packages and clutter the toc tree
    for app_module_detail in app_module_details:
        if app_module_detail[1]['Total members'] == 0:
            # see if it has a module contents section (not all will)
            filename = 'source/' + app_module_detail[0] + '.rst'
            infile = open(filename,'r+').readlines()
            if 'Module contents\n' in infile:
                inx_line = infile.index('Module contents\n')
                f = open(filename,'w')
                f.write("".join(infile[:inx_line]))
                f.close
                print ('Removing module contents from :' + filename)

             
def main(exclude_list,min_members_to_document,app):

    # return the initial list of files from sphinx-apidoc using a dummy run
    cmd = "sphinx-apidoc -f -e -n -o source ../" + app
    stream, err, errcode = process(cmd)
    
    if errcode == 0:
        print ('Sphinx-apidoc dummy list created')
    else:
        print ('Failed to create dummy list')
        return
    
    # pull out the files created
    app_list = shlex.split(stream)
    p = re.compile('(source/)(\S+).rst.')
    # create a list of modules
    app_modules = [x.group(2) for l in app_list for x in [p.match(l)] if x]

    # populate the contents of all the modules
    app_module_details = create_module_details(app_modules,exclude_list=exclude_list)

    # now create the rst files for real
    cmd = "sphinx-apidoc -f -e -o source ../" + app
    stream, err, errcode = process(cmd)
    if errcode == 0:
        print ('Base rst files built succcesfully')
    else:
        print ('Could not create the base rst files')

    # get user to agree to changes                 
    print ('The following modules have >= %0.f members :' % min_members_to_document)
    for app_module in app_module_details:
        if app_module[1]['Total members'] >= min_members_to_document:
            print (app_module[0])
            
    modify_files = raw_input('Proceed? y/n :')
    if modify_files.upper() == 'Y':
        update_modules(app_module_details,min_members_to_document)
    else:
        print ('No changes applied to modules')

    remove_module_contents = raw_input('Remove empty sections from intermediate pages? y/n :')
    if remove_module_contents.upper() == 'Y':
        remove_empty_module_contents(app_module_details)
    else:
        print ('No changes applied to intermediate pages')

if __name__ == '__main__':
    # if run from the command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exclude_list",
        nargs="*", 
        type=str,
        default=['module.rst'],  # default list if no arg value
        help = 'List of rst files to exclude from processing'
    )
    parser.add_argument(
            "min_members_to_document",
            type=int,
            default=5,
            help = 'Modules with less than this number of members will not get a summary table'
    )
    parser.add_argument(
            "app",
            type=str,
            default='',
            help = 'Name of the application to process'
    )
    args = parser.parse_args()
    main(args.exclude_list,args.min_members_to_document,args.app)
    
