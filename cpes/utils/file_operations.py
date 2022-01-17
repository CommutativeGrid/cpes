#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:01:04 2021

@author: hina
"""


import tempfile
import os
import requests


class TempfileFromUrl:
    """
    class to download a file from a url and save it to a temporary file
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
        except Exception:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()
