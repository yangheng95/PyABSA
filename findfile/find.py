# -*- coding: utf-8 -*-
# file: find.py
# time: 2021/8/4
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os
import re
import shutil
import warnings
from functools import reduce
from pathlib import Path
from typing import Union

from termcolor import colored

warnings.filterwarnings('once')


def accessible(search_path):
    try:
        os.listdir(search_path)
    except OSError:
        return False
    return True


def covert_path_sep(key_list):
    if isinstance(key_list, str):
        key_list = [key_list]
    new_key_list = []
    for key in key_list:
        if key and os.path.splitext(key):
            new_key_list.append(os.path.split(Path(key)))
        else:
            new_key_list.append(key)
    return key_list


def _find_files(search_path: Union[str, Path],
                key=None,
                exclude_key=None,
                use_regex=False,
                return_relative_path=True,
                **kwargs) -> list:
    '''
    'search_path': path to search
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path

    :return the files whose path contains the key(s)
    '''
    recursive = kwargs.pop('recursive', 5)
    if recursive is True:
        recursive = 5

    if not search_path:
        search_path = os.getcwd()

    res = []

    if not exclude_key:
        exclude_key = []
    if isinstance(exclude_key, str):
        exclude_key = [exclude_key]

    if isinstance(key, str):
        key = [key]

    if os.path.isfile(search_path):
        has_key = True
        for k in key:
            try:
                if use_regex:
                    if not re.findall(k.lower(), search_path.lower()):
                        has_key = False
                        break
                else:
                    if not k.lower() in search_path.lower():
                        has_key = False
                        break
            except re.error:
                warnings.warn('FindFile Warning --> Regex pattern error, using string-based search')
                if not k.lower() in search_path.lower():
                    has_key = False
                    break

        if has_key:
            if exclude_key:
                has_exclude_key = False
                for ex_key in exclude_key:
                    try:
                        if use_regex:
                            if re.findall(ex_key.lower(), search_path.lower()):
                                has_exclude_key = True
                                break
                        else:
                            if ex_key.lower() in search_path.lower():
                                has_exclude_key = True
                                break
                    except re.error:
                        warnings.warn('FindFile Warning ->> Regex pattern error, using string-based search')
                        if ex_key.lower() in search_path.lower():
                            has_exclude_key = True
                            break
                if not has_exclude_key:
                    res.append(search_path.replace(os.getcwd() + os.sep, '') if return_relative_path else search_path)
            else:
                res.append(search_path.replace(os.getcwd() + os.sep, '') if return_relative_path else search_path)

    if os.path.isdir(search_path) and accessible(search_path):
        items = os.listdir(search_path)
        for file in items:
            if recursive:
                res += _find_files(os.path.join(search_path, file),
                                   key=key,
                                   exclude_key=exclude_key,
                                   use_regex=use_regex,
                                   recursive=recursive - 1,
                                   return_relative_path=return_relative_path,
                                   **kwargs)

    return res


def find_file(search_path: Union[str, Path],
              and_key=None,
              exclude_key=None,
              use_regex=False,
              return_relative_path=True,
              return_deepest_path=False,
              disable_alert=False,
              **kwargs) -> str:
    '''
    'search_path': path to search
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path
    'return_deepest_path' True/False to return the deepest/shortest path if multiple targets found
    'disable_alert' no alert if multiple targets found

    :return the file whose path contains the key(s)
    '''
    '''
     'key': find a set of files/dirs whose absolute path contain the 'key'
     'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
     'recursive' integer, recursive search limit 
     'return_relative_path' return the relative path instead of absolute path
     :return the target files' path in current working directory
     '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_files(search_path=search_path,
                               key=key,
                               use_regex=use_regex,
                               exclude_key=exclude_key,
                               return_relative_path=return_relative_path,
                               return_deepest_path=return_deepest_path,
                               disable_alert=disable_alert,
                               **kwargs)
    else:
        res = _find_files(search_path=search_path,
                          key=key,
                          use_regex=use_regex,
                          exclude_key=exclude_key,
                          return_relative_path=return_relative_path,
                          return_deepest_path=return_deepest_path,
                          disable_alert=disable_alert,
                          **kwargs)

    if not return_deepest_path:
        _res = reduce(lambda x, y: x if len(x) < len(y) else y, res) if res else None
    else:
        _res = reduce(lambda x, y: x if len(x) > len(y) else y, res) if res else None
    if len(res) > 1 and not disable_alert:
        print('FindFile Warning --> multiple targets {} found, only return the {} path: <{}>'.format(res, 'deepest' if return_deepest_path else 'shortest', colored(_res, 'yellow')))
    return _res


def _find_dirs(search_path: Union[str, Path],
               key=None,
               exclude_key=None,
               use_regex=False,
               return_relative_path=True,
               **kwargs) -> list:
    '''
    'search_path': path to search
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit
    'return_relative_path' return the relative path instead of absolute path

    :return the dirs whose path contains the key(s)
    '''
    recursive = kwargs.pop('recursive', 5)
    if recursive is True:
        recursive = 5

    if not search_path:
        search_path = os.getcwd()

    res = []

    if not exclude_key:
        exclude_key = []
    if isinstance(exclude_key, str):
        exclude_key = [exclude_key]

    if isinstance(key, str):
        key = [key]

    if os.path.isdir(search_path):
        has_key = True
        for k in key:
            try:
                if use_regex:
                    if not re.findall(k.lower(), search_path.lower()):
                        has_key = False
                        break
                else:
                    if not k.lower() in search_path.lower():
                        has_key = False
                        break
            except re.error:
                warnings.warn('FindFile Warning --> Regex pattern error, using string-based search')
                if not k.lower() in search_path.lower():
                    has_key = False
                    break

        if has_key:
            if exclude_key:
                has_exclude_key = False
                for ex_key in exclude_key:
                    try:
                        if use_regex:
                            if re.findall(ex_key.lower(), search_path.lower()):
                                has_exclude_key = True
                                break
                        else:
                            if ex_key.lower() in search_path.lower():
                                has_exclude_key = True
                                break
                    except re.error:
                        warnings.warn('FindFile Warning --> Regex pattern error, using string-based search')
                        if ex_key.lower() in search_path.lower():
                            has_exclude_key = True
                            break
                if not has_exclude_key:
                    res.append(search_path.replace(os.getcwd() + os.sep, '') if return_relative_path else search_path)
            else:
                res.append(search_path.replace(os.getcwd() + os.sep, '') if return_relative_path else search_path)

    if os.path.isdir(search_path) and accessible(search_path):
        items = os.listdir(search_path)
        for file in items:
            if recursive:
                res += _find_dirs(os.path.join(search_path, file),
                                  key=key,
                                  exclude_key=exclude_key,
                                  use_regex=use_regex,
                                  recursive=recursive - 1,
                                  return_relative_path=return_relative_path,
                                  **kwargs)

    return res


def find_dir(search_path: Union[str, Path],
             and_key=None,
             exclude_key=None,
             use_regex=False,
             return_relative_path=True,
             return_deepest_path=False,
             disable_alert=False,
             **kwargs) -> str:
    '''
    'search_path': path to search
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path
    'return_deepest_path' True/False to return the deepest/shortest path if multiple targets found
    'disable_alert' no alert if multiple targets found

    :return the dir path
    '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_dirs(search_path=search_path,
                              key=key,
                              exclude_key=exclude_key,
                              use_regex=use_regex,
                              return_relative_path=return_relative_path,
                              return_deepest_path=return_deepest_path,
                              **kwargs)

    else:
        res += _find_dirs(search_path=search_path,
                          key=key,
                          exclude_key=exclude_key,
                          use_regex=use_regex,
                          return_relative_path=return_relative_path,
                          return_deepest_path=return_deepest_path,
                          **kwargs)

    if not return_deepest_path:
        _res = reduce(lambda x, y: x if len(x) < len(y) else y, res) if res else None
    else:
        _res = reduce(lambda x, y: x if len(x) > len(y) else y, res) if res else None
    if len(res) > 1 and not disable_alert:
        print('FindFile Warning --> multiple targets {} found, only return the {} path: <{}>'.format(res, 'deepest' if return_deepest_path else 'shortest', colored(_res, 'yellow')))
    return _res


def find_cwd_file(and_key=None,
                  exclude_key=None,
                  use_regex=False,
                  return_relative_path=True,
                  return_deepest_path=False,
                  disable_alert=False,
                  **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path
    'return_deepest_path' True/False to return the deepest/shortest path if multiple targets found
    'disable_alert' no alert if multiple targets found

    :return the target file path in current working directory
    '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_files(search_path=os.getcwd(),
                               key=key,
                               use_regex=use_regex,
                               exclude_key=exclude_key,
                               return_relative_path=return_relative_path,
                               return_deepest_path=return_deepest_path,
                               disable_alert=disable_alert,
                               **kwargs)
    else:
        res = _find_files(search_path=os.getcwd(),
                          key=key,
                          use_regex=use_regex,
                          exclude_key=exclude_key,
                          return_relative_path=return_relative_path,
                          return_deepest_path=return_deepest_path,
                          disable_alert=disable_alert,
                          **kwargs)

    if not return_deepest_path:
        _res = reduce(lambda x, y: x if len(x) < len(y) else y, res) if res else None
    else:
        _res = reduce(lambda x, y: x if len(x) > len(y) else y, res) if res else None
    if len(res) > 1 and not disable_alert:
        print('FindFile Warning --> multiple targets {} found, only return the {} path: <{}>'.format(res, 'deepest' if return_deepest_path else 'shortest', colored(_res, 'yellow')))
    return _res


def find_cwd_files(and_key=None,
                   exclude_key=None,
                   use_regex=False,
                   return_relative_path=True,
                   return_deepest_path=False,
                   disable_alert=False,
                   **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path

    :return the target files' path in current working directory
    '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_files(search_path=os.getcwd(),
                               key=key,
                               exclude_key=exclude_key,
                               use_regex=use_regex,
                               return_relative_path=return_relative_path,
                               return_deepest_path=return_deepest_path,
                               disable_alert=disable_alert,
                               **kwargs)
    else:
        res = _find_files(search_path=os.getcwd(),
                          key=key,
                          exclude_key=exclude_key,
                          use_regex=use_regex,
                          return_relative_path=return_relative_path,
                          return_deepest_path=return_deepest_path,
                          disable_alert=disable_alert,
                          **kwargs)
    return res


def find_files(search_path: Union[str, Path],
               and_key=None,
               exclude_key=None,
               use_regex=False,
               return_relative_path=True,
               return_deepest_path=False,
               disable_alert=False,
               **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path
    :return the target files' path in current working directory
    '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_files(search_path,
                               key=key,
                               exclude_key=exclude_key,
                               use_regex=use_regex,
                               return_relative_path=return_relative_path,
                               return_deepest_path=return_deepest_path,
                               disable_alert=disable_alert,
                               **kwargs)
    else:
        res = _find_files(search_path,
                          key=key,
                          exclude_key=exclude_key,
                          use_regex=use_regex,
                          return_relative_path=return_relative_path,
                          return_deepest_path=return_deepest_path,
                          disable_alert=disable_alert,
                          **kwargs)
    return res


def find_cwd_dir(and_key=None,
                 exclude_key=None,
                 use_regex=False,
                 return_relative_path=True,
                 return_deepest_path=False,
                 disable_alert=False,
                 **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key',
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path
    'return_deepest_path' True/False to return the deepest/shortest path if multiple targets found
    'disable_alert' no alert if multiple targets found

    :return the target dir path in current working directory
    '''
    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_dirs(search_path=os.getcwd(),
                              key=key,
                              exclude_key=exclude_key,
                              use_regex=use_regex,
                              return_relative_path=return_relative_path,
                              return_deepest_path=return_deepest_path,
                              disable_alert=disable_alert,
                              **kwargs)

    else:
        res = _find_dirs(search_path=os.getcwd(),
                         key=key,
                         exclude_key=exclude_key,
                         use_regex=use_regex,
                         return_relative_path=return_relative_path,
                         return_deepest_path=return_deepest_path,
                         disable_alert=disable_alert,
                         **kwargs)

    if not return_deepest_path:
        _res = reduce(lambda x, y: x if len(x) < len(y) else y, res) if res else None
    else:
        _res = reduce(lambda x, y: x if len(x) > len(y) else y, res) if res else None
    if len(res) > 1 and not disable_alert:
        print('FindFile Warning --> multiple targets {} found, only return the {} path: <{}>'.format(res, 'deepest' if return_deepest_path else 'shortest', colored(_res, 'yellow')))
    return _res


def find_cwd_dirs(and_key=None,
                  exclude_key=None,
                  use_regex=False,
                  return_relative_path=True,
                  return_deepest_path=False,
                  disable_alert=False,
                  **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path

    :return the target dirs' path in current working directory
    '''

    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_dirs(search_path=os.getcwd(),
                              key=key,
                              exclude_key=exclude_key,
                              use_regex=use_regex,
                              return_relative_path=return_relative_path,
                              return_deepest_path=return_deepest_path,
                              disable_alert=disable_alert,
                              **kwargs)

    else:
        res = _find_dirs(search_path=os.getcwd(),
                         key=key,
                         exclude_key=exclude_key,
                         use_regex=use_regex,
                         return_relative_path=return_relative_path,
                         return_deepest_path=return_deepest_path,
                         disable_alert=disable_alert,
                         **kwargs)

    return res


def find_dirs(search_path: Union[str, Path],
              and_key=None,
              exclude_key=None,
              use_regex=False,
              return_relative_path=True,
              return_deepest_path=False,
              disable_alert=False,
              **kwargs):
    '''
    'key': find a set of files/dirs whose absolute path contain the 'key'
    'exclude_key': file whose absolute path contains 'exclude_key' will be ignored
    'recursive' integer, recursive search limit 
    'return_relative_path' return the relative path instead of absolute path

    :return the target dirs' path in current working directory
    '''

    key = kwargs.pop('key', and_key)

    res = []
    or_key = kwargs.pop('or_key', '')
    if or_key and isinstance(or_key, str):
        or_key = [or_key]
    if or_key:
        if or_key and key:
            raise ValueError('The key and or_key arg are contradictory!')
        for key in or_key:
            res += _find_dirs(search_path,
                              key=key,
                              exclude_key=exclude_key,
                              use_regex=use_regex,
                              return_relative_path=return_relative_path,
                              return_deepest_path=return_deepest_path,
                              disable_alert=disable_alert,
                              **kwargs)

    else:
        res = _find_dirs(search_path,
                         key=key,
                         exclude_key=exclude_key,
                         use_regex=use_regex,
                         return_relative_path=return_relative_path,
                         return_deepest_path=return_deepest_path,
                         disable_alert=disable_alert,
                         **kwargs)

    return res


def rm_files(path=None, and_key=None, exclude_key=None, **kwargs):
    key = kwargs.pop('key', and_key)

    if not path:
        path = os.getcwd()

    or_key = kwargs.pop('or_key', '')
    if or_key and key:
        raise ValueError('The key and or_key arg are contradictory!')

    if key:
        fs = _find_files(search_path=path,
                         key=key,
                         exclude_key=exclude_key,
                         use_regex=kwargs.pop('use_regex', False),
                         recursive=kwargs.pop('recursive', 5),
                         return_relative_path=kwargs.pop('return_relative_path', False),
                         **kwargs)

        print(colored('FindFile Warning: Remove files {}'.format(fs), 'red'))

        for f in fs:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(colored('FindFile Warning: Remove file {} failed: {}'.format(f, e), 'red'))

    if or_key:
        fs = []
        for or_key in or_key:
            fs += _find_files(search_path=path,
                              key=or_key,
                              exclude_key=exclude_key,
                              use_regex=kwargs.pop('use_regex', False),

                              recursive=kwargs.pop('recursive', 5),
                              return_relative_path=kwargs.pop('return_relative_path', False),
                              **kwargs)

        print(colored('FindFile Warning: Remove files {}'.format(fs), 'red'))

        for f in fs:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(colored('FindFile Warning --> Remove file {} failed: {}'.format(f, e), 'red'))


def rm_dirs(path=None, and_key=None, exclude_key=None, **kwargs):
    key = kwargs.pop('key', and_key)

    if not path:
        path = os.getcwd()

    or_key = kwargs.pop('or_key', '')
    if or_key and key:
        raise ValueError('The key and or_key arg are contradictory!')

    if key:
        ds = _find_dirs(search_path=path,
                        key=key,
                        exclude_key=exclude_key,
                        use_regex=kwargs.pop('use_regex', False),
                        recursive=kwargs.pop('recursive', 5),
                        return_relative_path=kwargs.pop('return_relative_path', False),
                        **kwargs)

        print(colored('FindFile Warning: Remove dirs {}'.format(ds), 'red'))

        for d in ds:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(colored('FindFile Warning: Remove dir {} failed: {}'.format(d, e), 'red'))

    if or_key:
        ds = []
        for or_key in or_key:
            ds += _find_dirs(search_path=path,
                             key=or_key,
                             exclude_key=exclude_key,
                             use_regex=kwargs.pop('use_regex', False),
                             recursive=kwargs.pop('recursive', 5),
                             return_relative_path=kwargs.pop('return_relative_path', False),
                             **kwargs)

        print(colored('FindFile Warning: Remove dirs {}'.format(ds), 'red'))

        for d in ds:
            if os.path.exists(d):
                if os.path.exists(d):
                    try:
                        shutil.rmtree(d)
                    except Exception as e:
                        print(colored('FindFile Warning --> Remove dir {} failed: {}'.format(d, e), 'red'))


def rm_file(path=None, and_key=None, exclude_key=None, **kwargs):
    key = kwargs.pop('key', and_key)

    if not path:
        path = os.getcwd()

    or_key = kwargs.pop('or_key', '')
    if or_key and key:
        raise ValueError('The key and or_key arg are contradictory!')

    if key:
        fs = _find_files(search_path=path,
                         key=key,
                         exclude_key=exclude_key,
                         use_regex=kwargs.pop('use_regex', False),
                         recursive=kwargs.pop('recursive', 5),
                         return_relative_path=kwargs.pop('return_relative_path', False),
                         **kwargs)

        if len(fs) > 1:
            raise ValueError('Multi-files detected while removing single file.')

        print(colored('FindFile Warning: Remove file {}'.format(fs), 'red'))

        for f in fs:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(colored('FindFile Warning --> Remove file {} failed: {}'.format(f, e), 'red'))

    if or_key:
        fs = []
        for or_key in or_key:
            fs += _find_files(search_path=path,
                              key=or_key,
                              exclude_key=exclude_key,
                              use_regex=False,
                              recursive=kwargs.pop('recursive', 5),
                              return_relative_path=kwargs.pop('return_relative_path', False),
                              **kwargs)
        if len(fs) > 1:
            raise ValueError('Multi-files detected while removing single file.')

        print(colored('FindFile Warning --> Remove file {}'.format(fs), 'red'))

        for f in fs:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception as e:
                    print(colored('FindFile Warning: Remove file {} failed: {}'.format(f, e), 'red'))


def rm_dir(path=None, and_key=None, exclude_key=None, **kwargs):
    key = kwargs.pop('key', and_key)

    if not path:
        path = os.getcwd()

    or_key = kwargs.pop('or_key', '')
    if or_key and key:
        raise ValueError('The key and or_key arg are contradictory!')

    if key:
        ds = _find_dirs(search_path=path,
                        key=key,
                        exclude_key=exclude_key,
                        use_regex=kwargs.pop('use_regex', False),
                        recursive=kwargs.pop('recursive', 5),
                        return_relative_path=kwargs.pop('return_relative_path', False),
                        **kwargs)

        if len(ds) > 1:
            raise ValueError('Multi-dirs detected while removing single file.')

        print(colored('FindFile Warning: Remove dirs {}'.format(ds), 'red'))

        for d in ds:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(colored('FindFile Warning --> Remove dirs {} failed: {}'.format(d, e), 'red'))

    if or_key:
        ds = []
        for or_key in or_key:
            ds += _find_dirs(search_path=path,
                             key=or_key,
                             exclude_key=exclude_key,
                             use_regex=kwargs.pop('use_regex', False),
                             recursive=kwargs.pop('recursive', 5),
                             return_relative_path=kwargs.pop('return_relative_path', False),
                             **kwargs)

        if len(ds) > 1:
            raise ValueError('Multi-dirs detected while removing single file.')

        print(colored('FindFile Warning: Remove dirs {}'.format(ds), 'red'))

        for d in ds:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(colored('FindFile Warning --> Remove dirs {} failed: {}'.format(d, e), 'red'))


def rm_cwd_file(and_key=None, exclude_key=None, **kwargs):
    rm_file(os.getcwd(), and_key, exclude_key, **kwargs)


def rm_cwd_files(and_key=None, exclude_key=None, **kwargs):
    rm_files(os.getcwd(), and_key, exclude_key, **kwargs)


def rm_cwd_dir(and_key=None, exclude_key=None, **kwargs):
    rm_dir(os.getcwd(), and_key, exclude_key, **kwargs)


def rm_cwd_dirs(and_key=None, exclude_key=None, **kwargs):
    rm_dirs(os.getcwd(), and_key, exclude_key, **kwargs)
