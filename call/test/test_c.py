
#   Test accessing C code from Python

from call.c import load_library


def test_call_c_load_library():
    load_library()
