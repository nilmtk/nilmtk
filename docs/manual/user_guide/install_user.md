# Installing NILMTK

We recommend using [Anaconda](https://store.continuum.io/cshop/anaconda/), which bundles togther most of the required packages. Since many packages are deprecating  Python 2, we are now recommending installing Python 3.6. For the time being, Python 2.7 is still supported but that may change in the near future.

After Anaconda has been installed, open up the terminal (Unix) or Anaconda prompt(Windows) :

1.  NILMTK should work fine in the base environment, but we recommend creating a new environment where NILMTK and related dependecies are installed.

	```bash
	conda create --name nilmtk-env 
	```

2. Add conda-forge to list of channels to be searched for packages.
	```bash
	conda config --add channels conda-forge
	```

2. Activate the new *nilmtk-env* environment.

	```bash
	conda activate nilmtk-env
	```

3. Install the NILMTK package

	```bash
	conda install -c nilmtk nilmtk
	```

4. The installed package import for python/ ipython can be  tested in the terminal using the following command:
	```bash
	python -c "import nilmtk"
	```
	or	
	```bash
	ipython -c "import nilmtk"
	```
	> Note: This might show DepreciationWarning due to the *imp* module. That will be fixed in a future release.

	* Alternatively, you can also run your IDE in *nilmtk-env* from: Anaconda Navigator > "Applications on" dropdown > nilmtk-env
	To check the current environment variables,

		```python
		import sys
		print(sys.executable)
		print(sys.version)
		```
		You will see an output similar to:
		```
		/home/ayush/anaconda3/envs/nilmtk-env/bin/python
		3.6.7 | packaged by conda-forge | (default, Feb 28 2019, 09:07:38) 
		[GCC 7.3.0]
		```
5. Run your python IDE from this environment, for example:

	```bash
	jupyter notebook
	```
	or

	```bash
	spyder
	```

6. Import NILMTK in the IDE:

	```python
	import nilmtk
	```
	The package modules can now be used.
7. To deactivate this environment,

	```bash
	conda deactivate
	```