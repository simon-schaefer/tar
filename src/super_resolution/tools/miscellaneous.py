#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Collection of miscellaneous helper functions. 
# =============================================================================
import os

def print_header() -> None: 
    header_file = os.environ['SR_PROJECT_SCRIPTS_PATH'] + "/header.bash"
    os.system("bash " + header_file)