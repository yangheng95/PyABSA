# -*- coding: utf-8 -*-
# file: wrappers.py
# time: 17:58 2023/1/7
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.


from multiprocessing import Process, Pipe
from eventlet import spawn, sleep
from functools import wraps

import traceback


def read_msg_from_son_proc(f_conn):
    while True:
        msg_poll = f_conn.poll()
        if msg_poll:
            msg = f_conn.recv()
            if msg:
                raise Exception(msg)
            break
        sleep(0.1)


# func: wrap ,func run in other proc ,
#        and read Except from son proc
#
# !!! if you want use it not in dead loop ,
#    please consider , wait for this func
#
#  beacause this is a async wrapper
#  so it can't wait for return, it just can get Except in son process
def run_in_async_process(f):
    @wraps(f)
    def wrapper(*a, **ka):
        # must be have a conn to send message to father
        def wrap_f(*a, **ka):
            msg = ""
            conn = ka.pop("conn")
            try:
                f(*a, **ka)
            except:
                msg = "{}\nsome error catch by son proc".format(traceback.format_exc())
            finally:
                conn.send(msg)
                exit(0)

        f_conn, s_conn = Pipe()
        ka.update({"conn": s_conn})
        p = Process(target=wrap_f, args=a, kwargs=ka)
        p.start()
        x = spawn(read_msg_from_son_proc, f_conn)
        x.wait()

    return wrapper
