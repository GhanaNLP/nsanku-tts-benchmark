"""Fast YAML reading.

The per-language results files carry every scored clip, so each is ~0.5-1 MB and a
sync parses all of them several times. PyYAML's pure-Python loader takes ~1.7 s per
file; libyaml's C loader about 0.2 s. Falls back to the pure-Python loader where
libyaml is not installed.
"""

import yaml

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load(source):
    """Parse YAML from a string or an open file."""
    return yaml.load(source, Loader=Loader)
