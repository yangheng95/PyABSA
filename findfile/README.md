# findfile - simplified solution of FileNotFoundError

[![Downloads](https://pepy.tech/badge/findfile)](https://pepy.tech/project/findfile)
[![Downloads](https://pepy.tech/badge/findfile/month)](https://pepy.tech/project/findfile)
[![Downloads](https://pepy.tech/badge/findfile/week)](https://pepy.tech/project/findfile)

![PyPI - Python Version](https://img.shields.io/badge/python-3.6-blue.svg)
[![PyPI](https://img.shields.io/pypi/v/findfile)](https://pypi.org/project/findfile/)
[![PyPI_downloads](https://img.shields.io/pypi/dm/findfile)](https://pypi.org/project/findfile/)
![Repo Size](https://img.shields.io/github/repo-size/yangheng95/findfile)

This is a package for you to locate your target file(s)/dir(s) easily.

# Usage

## Install

```
pip install findfile
```

## ready to use

If you have been bothered by FileNotFoundError while the file does exist but misplaced, you can call

```
from findfile import find_file, find_files, find_dir, find_dirs

search_path = './'

key = ['target', '.txt']  # str or list, the files whose absolute path contain all the keys in the key are the target files

exclude_key = ['dev', '.ignore']  # str or list, the files whose absolute path contain any exclude key are ignored

target_file = find_file(search_path, key, exclude_key, recursive=False)   # return the first target file, recursive means to search in all subdirectories

target_files = find_files(search_path, key, exclude_key, recursive=True)   # return all the target files, only the first param are required

target_dir = find_dir(search_path, key, exclude_key)  # search directory instead of file

target_dirs = find_dirs(search_path, key, exclude_key)  # search directories 


# rm_file(key=['findfile', 'lib'])
# rm_files(key=['findfile', 'lib'])
# rm_dir(key=['dist'])
# rm_dirs(key=['dist'])
# rm_dir(or_key=['dist', 'egg'])
rm_dirs(or_key=['dist', 'egg'])
