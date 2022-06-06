# Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function
from ctypes import *

lib = cdll.LoadLibrary("./libs/disease.so")

# describe and invoke Add()
lib.Add.argtypes = [c_longlong, c_longlong]
lib.Add.restype = c_longlong
print("disease.Add(12,99) = %d" % lib.Add(12,99))

# describe and invoke Cosine()
lib.Cosine.argtypes = [c_double]
lib.Cosine.restype = c_double
print("disease.Cosine(1) = %f" % lib.Cosine(1))

# define class GoSlice to map to:
# C type struct { void *data; GoInt len; GoInt cap; }
class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_void_p)), ("len", c_longlong), ("cap", c_longlong)]

nums = GoSlice((c_void_p * 5)(74, 4, 122, 9, 12), 5, 5) 

# call Sort
lib.Sort.argtypes = [GoSlice]
lib.Sort.restype = None
lib.Sort(nums)
print("disease.Sort(74,4,122,9,12) = %s" % [nums.data[i] for i in range(nums.len)])

# define class GoString to map:
# C type struct { const char *p; GoInt n; }
class GoString(Structure):
    _fields_ = [("p", c_char_p), ("n", c_longlong)]

# describe and call Log()
lib.Log.argtypes = [GoString]
lib.Log.restype = c_longlong
msg = GoString(b"Hello Bhojpur Disease!", 13)
print("log id %d"% lib.Log(msg))
