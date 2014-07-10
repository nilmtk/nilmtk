Contributing
------------

Contributing documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, we clone the main `nilmtk` repository::

	cd ~
	git clone https://github.com/nilmtk/nilmtk/

Next, we create a folder named `nilmtkdocs`::

	cd ~
	mkdir nilmtkdocs

Inside `nilmtkdocs`, we create a `html` folder, within which we clone the `nilmtk` repository::

	cd nilmtkdocs
	git clone https://github.com/nilmtk/nilmtk/ html

Setting up remote for `gh-pages` branch inside `html` folder::
	
	cd html
	git checkout -b gh-pages remotes/origin/gh-pages

Now, move back to the main repository::

	cd ~/nilmtk

As usual make changes to .rst files in `~/nilmtk/docs/source` to contribute to documentation. 

Build your documentation::

	cd ~/nilmtk/docs
	make html

Now, built documentation should lie in `~/nilmtkdocs` directory. You may use your browser to check if everything got built properly.

Commit your changes and push. These changes are reflected in the master
branch of `nilmtk` a these are changes to the source. ::
	
	git commit -m "your message"
	git push

Next, move to the `nilmtkdocs/html` folder::

	cd ~/nilmtkdocs/html

Check branch::

	git branch

It should be `gh-pages`. Now, this `gh-pages` branch should contain your newly built docs. Push these to `gh-pages` branch and see them appear magically on the
official documentation site::

	git commit -a -m "rebuilt docs"
	git push origin gh-pages

Thats all! Please allow a time upto 10 minutes for your new docs to appear on the official documentation site.
Thanks to the wonderful tutorial here_. 

.. _here: https://github.com/daler/sphinxdoc-test


sphinx-apidoc
=============

We use `sphinx-apidoc <http://sphinx-doc.org/man/sphinx-apidoc.html>`_
to automatically generate out API pages from the code.  Run this code
from the root nilmtk directory to regenerate the docs::

    sphinx-apidoc -f -o docs nilmtk
