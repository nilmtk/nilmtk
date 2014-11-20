Contributing
------------

Contributing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, we clone the main `nilmtk` repository::

	cd ~
	git clone https://github.com/nilmtk/nilmtk/

`nilmtksite` would contain all the content which gets uploaded to nilmtk.github.io, including sphinx content.::

	git clone https://github.com/nilmtk/nilmtk.github.io nilmtksite

Now, move back to the main repository::

	cd ~/nilmtk

As usual make changes to .rst files in `~/nilmtk/docs/source` to contribute to documentation. 

Build your documentation::

	cd ~/nilmtk/docs
	make html

Now, built documentation should lie in `~/nilmtksite/html` directory. You may use your browser to check if everything got built properly.

Commit your changes and push to nilmtk. And then commit all your changes to nilmtksite. Thats all!



Using IPython notebooks
~~~~~~~~~~~~~~~~~~~~~~~

Follow `this <http://sphinx-ipynb.readthedocs.org/en/latest/howto.html>`_ guide!



sphinx-apidoc
=============

We use `sphinx-apidoc <http://sphinx-doc.org/man/sphinx-apidoc.html>`_
to automatically generate API pages from the code.  Run this code
from the root nilmtk directory to regenerate the docs::

    /nilmtk$ sphinx-apidoc -f -o docs/source nilmtk
