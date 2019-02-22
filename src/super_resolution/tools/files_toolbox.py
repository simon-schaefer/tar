#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of tool function for handling files and directories.
# =============================================================================
import os
import typing

def load_files(directory: str, extension: str) -> typing.List[str]: 
    ''' Returning list of files in given directory having the 
    given file extension (e.g. extension = "png"). The files are 
    returned by their absolute paths. '''
    files = []
    for filename in os.listdir(directory):
        if filename.endswith("." + extension): 
            files.append(os.path.join(directory, filename))
    return files

