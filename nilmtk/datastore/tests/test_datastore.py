from pathlib import Path

import yaml

from nilmtk.datastore.datastore import join_key, write_yaml_to_file

GENERIC_YAML = """# Scalars
string: "Hello, YAML!"
integer: 42
float: 3.14159
boolean: true
null_value: null

# Sequences (Lists)
list:
- "Item 1"
- 2
- true
- 3.14
- null
- { key: "value" }

# Mappings (Dictionaries)
mapping:
key1: "value1"
key2: 123
key3:
    nested_key: "Nested value"
key4: [1, 2, 3]

# Mixed Types
mixed:
- { name: "Alice", age: 30, married: false }
- [ "nested", "list", 3.14 ]
- null
- "Just a string"

# Timestamp
timestamp: 2024-02-16T12:34:56Z
"""


def test_write_yaml_to_file(tmp_path: Path):
    tmp_file = Path(tmp_path, "test.yml")
    left = yaml.load(GENERIC_YAML, yaml.FullLoader)
    write_yaml_to_file(tmp_file, left)
    with open(tmp_file, "r") as stream:
        assert left == yaml.load(stream, yaml.FullLoader)


def test_join_key():
    ret = join_key("building1", "elec", "meter1")
    assert ret == "/building1/elec/meter1"

    ret = join_key("/")
    assert ret == "/"

    ret = join_key("")
    assert ret == "/"
