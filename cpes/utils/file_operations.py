import tempfile
import os
import requests


class TempfileFromUrl:
    """
    A context manager class to download a file from a URL and save it to a temporary file. 
    The file is automatically deleted when the context is exited.

    Parameters
    ----------
    url : str
        The URL from which the file will be downloaded.

    Attributes
    ----------
    url : str
        The URL provided for downloading the file.
    file : tempfile.NamedTemporaryFile
        The temporary file object where the downloaded content is stored.

    Methods
    -------
    path()
        Returns the path of the temporary file.
    __enter__()
        Context manager entry method. Downloads the file and prepares the temporary file.
    __exit__(exc_type, exc_value, traceback)
        Context manager exit method. Cleans up by closing and deleting the temporary file.
    __del__()
        Destructor method to ensure the temporary file is closed and deleted.

    Examples
    --------
    >>> with TempfileFromUrl('http://example.com/file') as temp_file:
    ...     for line in temp_file:
    ...         print(line)
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        response.raise_for_status()
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.file.write(response.content)

    def path(self):
        return self.file.name

    def __enter__(self):
        print(f"Downloading file from {self.url}.")
        self.file.seek(0)
        return self.file

    def __del__(self):
        self.file.close()
        try:
            os.remove(self.file.name)
        except BaseException as e:
            print(e)
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()
