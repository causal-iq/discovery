#
#   Wrapper to call C code from python
#

import ctypes
import ctypes.util


def load_library():
    path = ctypes.util.find_library("msvcrt")
    print("\nmsvcrt library found at {}".format(path))

    # path = 'call/C/mylib.dll'
    libc = ctypes.CDLL(path, winmode=0)
    print('Library handle type: {}'.format(type(libc)))

    libc.puts(b"Using C puts function to write this out")

    # Shred object filr that works was generated using:
    # gcc -shared -Wl,-soname,testlib -o testlib.so -fPIC testlib.c
    try:
        lib = ctypes.CDLL('C:\\dev\\git\\sharedplanet\\' +
                          'bnbench\\call\\c\\testlib.so')
    except OSError as e:
        print("Error:", e)

    print('\ntestlib.so library handle is: {}'.format(lib))

    print('\nCalling testlib.myprint() ...\n')
    lib.myprint()

    print('\nYay!')
