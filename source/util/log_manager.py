#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:13:34 2023

@author: placais
"""
import os
import logging.config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'console': {
            'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        },
        'file': {
            'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'console',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file': {
            'level': 'INFO',
            'formatter': 'file',
            'class': 'logging.FileHandler',
            'encoding': 'utf-8',
            'filename': 'lightwin.log',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False
        },
        'default': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

def set_up(project_dir: str = None):
    """Set up the logger."""
    if project_dir is not None:
        LOGGING_CONFIG['handlers']['file']['filename'] = os.path.join(
            project_dir, LOGGING_CONFIG['handlers']['file']['filename'])

    logging.config.dictConfig(LOGGING_CONFIG)



# def set_up(filepath: str) -> None:
#     """Set up the global log parameters."""
#     common_fmt = "dict_c[col] %(asctime)s - %(levelname)s - dict_c[normal] (message)s"
#     fmt_info = "\x1b[36m %(asctime)s - %(levelname)s - \x1b[0m %(message)s"

#     dict_c = {
#         'info': '\x1b[36m',     # cyan
#         'warning': '\x1b[34m',     # blue
#         'error': '\x1b[35m',  # magenta
#         'critical': '\x1B[31m',      # red
#         'normal': '\x1b[0m',    # normal
#         'green': '\x1b[32m',    # green
#     }

#     log.basicConfig(
#         # filename=os.path.join(filepath, 'log.log'),
#         level=log.INFO,
#         format="\x1B[31m %(asctime)s - %(levelname)s - \x1b[0m %(message)s")

# class CustomFormatter(logging.Formatter):
#     print('create class')
#     grey = "\x1b[38;20m"
#     yellow = "\x1b[33;20m"
#     red = "\x1b[31;20m"
#     bold_red = "\x1b[31;1m"
#     reset = "\x1b[0m"
#     format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

#     FORMATS = {
#         logging.DEBUG: grey + format + reset,
#         logging.INFO: grey + format + reset,
#         logging.WARNING: yellow + format + reset,
#         logging.ERROR: red + format + reset,
#         logging.CRITICAL: bold_red + format + reset
#     }

#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)


# def set_up(filepath: str):
#     # create logger with 'spam_application'
#     logger = logging.getLogger("My_app")
#     logger.setLevel(logging.INFO)

#     # create console handler with a higher log level
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)

#     ch.setFormatter(CustomFormatter())

#     logger.addHandler(ch)
#     return logger
