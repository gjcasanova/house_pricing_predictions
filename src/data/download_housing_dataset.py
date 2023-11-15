"""Download and extract the housing dataset from the external source."""

# Third party
import tarfile
from urllib import request

# Local modules
from src.config.settings import HOUSING_DATA_URL, PROJECT_DATA_PATH

DEFAULT_DOWNLOAD_URL = HOUSING_DATA_URL
DEFAULT_DOWNLOAD_DESTINY_PATH = PROJECT_DATA_PATH / 'raw/housing/housing.tgz'
DEFAULT_EXTRACTION_DESTINY_PATH = PROJECT_DATA_PATH / 'raw/housing'


def fetch_housing_data(download_url=DEFAULT_DOWNLOAD_URL, destiny_path=DEFAULT_DOWNLOAD_DESTINY_PATH):
    """Fetch and save the data from a remote origin."""
    destiny_path.parent.mkdir(exist_ok=True, parents=True)
    request.urlretrieve(download_url, destiny_path)


def extract_housing_data(file_path=DEFAULT_DOWNLOAD_DESTINY_PATH,
                         destiny_path=DEFAULT_EXTRACTION_DESTINY_PATH):
    """Extract content from a .tgz file and save its content."""
    downloaded_file = tarfile.open(file_path)
    downloaded_file.extractall(path=destiny_path)
    downloaded_file.close()


def main():
    """Fetch, extract and save the housing dataset."""
    fetch_housing_data()
    extract_housing_data()


if __name__ == '__main__':
    main()
