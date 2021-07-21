# Tests with DEDDIAG
[Scientific Publication](https://doi.org/10.1038/s41597-021-00963-2)  
[Dataset Download](https://doi.org/10.6084/m9.figshare.13615073)  
Only the data from house 8 will be converted. 

### Prerequirements

- Installed package `deddiag-loader`

### Usage

**1. DEDDIAG-database**  
- Download `house_08.zip`, `import.sh`, `create_tables_0.sql`, `create_tables_1.sql`  from [Figshare](https://doi.org/10.6084/m9.figshare.13615073).
- Start a docker-container with the DEDDIAG-database: 
`docker run -d --name deddiagdb -p 127.0.0.1:5432:5432 -e POSTGRES_PASSWORD=password postgres`
- Unpack `house_08.zip`
- Run `import.sh`

A full description for importing the dataset can be found in the [README.md](https://figshare.com/articles/dataset/DEDDIAG_a_domestic_electricity_demand_dataset_of_individual_appliances_in_Germany/13615073/1?file=26191907)

**2. Convert data**
- Convert data 
```python
from nilmtk.dataset_converters.deddiag.convert_deddiag import convert_deddiag
from deddiag_loader import Connection

# Connction to the DEDDIAG-database
connection = Connection(host="localhost", port="5432", db_name="postgres", user="postgres", password="password")
# Convert data and save
convert_deddiag(connection, '/data/deddiag.h5')
```

- Load data in NILMTK
```python
# Load data in NILMTK
from nilmtk import DataSet
deddiag = DataSet('/data/deddiag.h5')
building = 8
elec = deddiag.buildings[building].elec
```
