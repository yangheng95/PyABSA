# -*- coding: utf-8 -*-
# file: checkpoint_template.py
# time: 02/11/2022 15:44
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# GScholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# ResearchGate: https://www.researchgate.net/profile/Heng-Yang-17/research
# Copyright (C) 2022. All Rights Reserved.

from argparse import Namespace
from pyabsa.framework.configuration_class.config_verification import config_check
from pyabsa.utils.pyabsa_utils import fprint


class ConfigManager(Namespace):
    def __init__(self, args=None, **kwargs):
        """
        The ConfigManager is a subclass of argparse.Namespace and based on a parameter dict.
        It also counts the call-frequency of each parameter.

        :param args: A parameter dict.
        :param kwargs: Same params as Namespace.
        """
        if not args:
            args = {}
        super().__init__(**kwargs)

        if isinstance(args, Namespace):
            self.args = vars(args)
            self.args_call_count = {arg: 0 for arg in vars(args)}
        else:
            self.args = args
            self.args_call_count = {arg: 0 for arg in args}

    def __getattribute__(self, arg_name):
        """
        Get the value of an argument and increment its call count.

        :param arg_name: The name of the argument.
        :return: The value of the argument.
        """
        if arg_name == "args" or arg_name == "args_call_count":
            return super().__getattribute__(arg_name)
        try:
            value = super().__getattribute__("args")[arg_name]
            args_call_count = super().__getattribute__("args_call_count")
            args_call_count[arg_name] += 1
            super().__setattr__("args_call_count", args_call_count)
            return value

        except Exception as e:
            return super().__getattribute__(arg_name)

    def __setattr__(self, arg_name, value):
        """
        Set the value of an argument and add it to the argument dict and call count dict.

        :param arg_name: The name of the argument.
        :param value: The value of the argument.
        """
        if arg_name == "args" or arg_name == "args_call_count":
            super().__setattr__(arg_name, value)
            return
        try:
            args = super().__getattribute__("args")
            args[arg_name] = value
            super().__setattr__("args", args)
            args_call_count = super().__getattribute__("args_call_count")

            if arg_name in args_call_count:
                super().__setattr__("args_call_count", args_call_count)

            else:
                args_call_count[arg_name] = 0
                super().__setattr__("args_call_count", args_call_count)

        except Exception as e:
            super().__setattr__(arg_name, value)

    def get(self, key, default=None):
        """
        Get the value of a key from the parameter dict. If the key is found, increment its call frequency.
        :param key: The key to look for in the parameter dict.
        :param default: The default value to return if the key is not found.
        :return: The value of the key in the parameter dict, or the default value if the key is not found.
        """
        if key in self.args_call_count:
            self.args_call_count[key] += 1
        return self.args.get(key, default)

    def update(self, *args, **kwargs):
        """
        Update the parameter dict with the given arguments and keyword arguments, and check if the updated configuration is valid.
        :param args: Positional arguments to update the parameter dict.
        :param kwargs: Keyword arguments to update the parameter dict.
        """
        self.args.update(*args, **kwargs)
        config_check(self.args)

    def pop(self, *args):
        """
        Pop a value from the parameter dict.
        :param args: Arguments to pop from the parameter dict.
        :return: The value popped from the parameter dict.
        """
        return self.args.pop(*args)

    def keys(self):
        """
        Get a list of all keys in the parameter dict.
        :return: A list of all keys in the parameter dict.
        """
        return self.args.keys()

    def values(self):
        """
        Get a list of all values in the parameter dict.
        :return: A list of all values in the parameter dict.
        """
        return self.args.values()

    def items(self):
        """
        Get a list of all key-value pairs in the parameter dict.
        :return: A list of all key-value pairs in the parameter dict.
        """
        return self.args.items()

    def __str__(self):
        """
        Get a string representation of the parameter dict.
        :return: A string representation of the parameter dict.
        """
        return str(self.args)

    def __repr__(self):
        """
        Return the string representation of the parameter dict.
        """
        return repr(self.args)

    def __len__(self):
        """
        Return the number of items in the parameter dict.
        """
        return len(self.args)

    def __iter__(self):
        """
        Return an iterator over the keys of the parameter dict.
        """
        return iter(self.args)

    def __contains__(self, item):
        """
        Check if the given item is in the parameter dict.
        :param item: The item to check.
        :return: True if the item is in the parameter dict, False otherwise.
        """
        return item in self.args

    def __getitem__(self, item):
        """
        Get the value of a key from the parameter dict.
        :param item: The key to look for in the parameter dict.
        :return: The value of the key in the parameter dict.
        """
        return self.args[item]

    def __setitem__(self, key, value):
        """
        Set the value of a key in the parameter dict. Also set the call frequency of the key to 0 and check if the updated
        configuration is valid.
        :param key: The key to set the value for in the parameter dict.
        :param value: The value to set for the key in the parameter dict.
        """
        self.args[key] = value
        self.args_call_count[key] = 0
        config_check(self.args)

    def __delitem__(self, key):
        """
        Delete a key-value pair from the parameter dict and check if the updated configuration is valid.
        :param key: The key to delete from the parameter dict.
        """
        del self.args[key]
        config_check(self.args)

    def __eq__(self, other):
        """
        Check if the parameter dict is equal to another object.
        :param other: The other object to compare with the parameter dict.
        :return: True if the parameter dict is equal to the other object, False otherwise.
        """
        return self.args == other

    def __ne__(self, other):
        """
        Check if the parameter dict is not equal to another object.
        :param other: The other object to compare with the parameter dict.
        :return: True if the parameter dict is not equal to the other object, False otherwise.
        """
        return self.args != other


if __name__ == "__main__":  # test
    config = ConfigManager({"a": 1, "b": 2})
    config.a = 2
    config.b = 3
    config.c = 4
    fprint(config.a)
    fprint(config.b)
    fprint(config.c)
    fprint(config.args_call_count)
