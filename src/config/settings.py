"""Save settings for the project like constants and other general values."""

# Third party
from pathlib import Path

# Paths
PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent
PROJECT_DATA_PATH = PROJECT_ROOT_PATH / 'data'

# Remote sources
HOUSING_REPOSITORY_URL = 'https://raw.githubusercontent.com/ageron/handson-ml2/master'
HOUSING_DATA_URL = f'{HOUSING_REPOSITORY_URL}/datasets/housing/housing.tgz'
