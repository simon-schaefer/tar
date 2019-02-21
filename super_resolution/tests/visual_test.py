#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Simon Schaefer
# Description : Helper functions for visual test cases. 
# =============================================================================

def get_input(text):
    return input(text)

def visual_test():
    ans = get_input('output satisfying ? (y/n)')
    if ans == 'y':
        print("output passed visual test !")
        return True
    else:
        print("output failed visual test !")
        return False
    
