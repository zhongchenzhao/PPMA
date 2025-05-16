#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil, glob
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def save_codes(save_path):
    # copy code
    save_path = os.path.join(save_path, 'source_codes')
    os.makedirs(save_path, exist_ok=True)
    source = glob.glob("*.py")
    source += glob.glob("*/*.py")
    source += glob.glob("*/*/*.py")
    source += glob.glob("*/*/*/*.py")
    source += glob.glob("*/*/*/*/*.py")
    source += glob.glob("*/*/*/*/*/*.py")
    for file in source:
        name = file.split("/")[0]
        if name == file:
            shutil.copy(file, save_path)
        else:
            folder = "/".join(file.split("/")[:-1])
            if 'source_codes' in folder or 'exp' in folder:
                pass        # skip 'source_codes' folder
            else:
                os.makedirs(os.path.join(save_path, folder), exist_ok=True)
                shutil.copy(file, os.path.join(save_path, folder))


def save_python_command(save_path):
    def read_python_command():
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.splitlines()
            python_processes = ["python" +line.split("python")[-1] for line in lines if 'python' in line]
            if python_processes:
                state = 0
                return state, python_processes[-1]
            else:
                state = -2
                return state, ""

        except Exception as e:
            state = -1
            print(f"Error: An error occurred: {e}")
            return state, ""

    state, commond = read_python_command()
    save_path = os.path.join(save_path, 'python_command.sh')
    with open(save_path, "w") as f:
        f.write(commond)


if __name__ == "__main__":
    pass
