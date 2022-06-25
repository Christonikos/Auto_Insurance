#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module that includes:
    1. generic, re-usable functions/classes (i.e: class to create dirs)
    2. hyperparameters (i.e: the random seed)
    3. The logging configuration



@author: Christos
"""
# =============================================================================
# MODULES & ALLIASES
# =============================================================================
import os
import logging
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# alliases
join = os.path.join


class FetchPaths():
    '''
    Simple class to get the paths. All paths are returned as str type.
    Attributes:
        1. projects_path
        2. project name
    '''

    def __init__(self, projects_path, project_name,):
        self.projects_path = projects_path
        self.project_name = project_name

    def to_project(self):
        '''
        Returns the project path.
        '''
        return join(self.projects_path, self.project_name,)

    def to_data_raw(self):
        '''
        Returns the path where the raw data are stored.
        '''
        return join(self.projects_path, self.project_name, 'data', 'raw')

    def to_data_preprocessed(self):
        '''
        Returns the path where the  preprocessed data are stored.
        '''
        return join(self.projects_path, self.project_name, 'data', 'preprocessed')

    def to_images(self):
        '''
        Returns the path where the images are stored as .png files.
        '''
        return join(self.projects_path, self.project_name, 'images')

    def to_logs(self):
        '''
        Returns the path where the logs are stored as .log files.
        '''
        return join(self.projects_path, self.project_name, 'logs')

    def to_results(self):
        '''
        Returns the path where the predicted classes are stored as .npy files.
        '''
        return join(self.projects_path, self.project_name, 'results')

    def to_params(self):
        '''
        Returns the path where the params are stored as .pkl files.
        '''
        return join(self.projects_path, self.project_name, 'params')


    def __str__(self):
        return f'Project: {self.project_name}'


# =============================================================================
# PROJECT ATTRIBUTES
# =============================================================================
# The PROJECTS_PATH is where the code for all running projects are stored
PROJECTS_PATH = '/Users/christoszacharopoulos/projects/'
PROJECT_NAME = 'insurance'


# =========================================================ß====================
# SET UP THE LOGGING CONFIGURATION
# =============================================================================
log_filename=join(FetchPaths(PROJECTS_PATH,PROJECT_NAME).to_logs(),
                  'results.log')

# set up the logging file
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')

# unicode characters to log success and errors
error = 4 * '\u274C'
success = 4 * '\u2705'

# =============================================================================
# GLOBALS & HYPERPARAMS
# =============================================================================
random_state = 42

# LightGBM hyperparameters
param_test = {
    "num_leaves": sp_randint(6, 50),
    "min_child_samples": sp_randint(100, 500),
    "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    "subsample": sp_uniform(loc=0.2, scale=0.8),
    "colsample_bytree": sp_uniform(loc=0.4, scale=0.6),
    "reg_alpha": [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    "reg_lambda": [0, 1e-1, 1, 5, 10, 20, 50, 100],
}


