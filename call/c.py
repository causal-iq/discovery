#
#   Wrapper to call C code from python
#

from os import getcwd
import ctypes
import ctypes.util


def load_library():
    path = ctypes.util.find_library("msvcrt")
    print("\nmsvcrt library found at {}".format(path))

    # path = 'call/C/mylib.dll'
    libc = ctypes.CDLL(path, winmode=0)
    print('Library handle type: {}'.format(type(libc)))

    libc.puts(b"Using C puts function to write this out")

    # Load shared object library, which was previously generated using:
    # gcc -shared -Wl,-soname,testlib -o testlib.so -fPIC testlib.c

    print(getcwd())
    path = (getcwd() + "\\call\\C\\testlib.so").replace("\\", "\\\\")
    print(path)
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        print("Error:", e)

    print('\ntestlib.so library handle is: {}'.format(lib))

    # check we can print using "myprint"  in the loaded .so library
    print('\nCalling testlib.myprint() ...\n')
    lib.myprint()

    print('\nYay!')
