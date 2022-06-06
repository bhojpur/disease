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
import sys
from cffi import FFI

is_64b = sys.maxsize > 2**32

ffi = FFI()
if is_64b: ffi.cdef("typedef long GoInt;\n")
else:      ffi.cdef("typedef int GoInt;\n")

ffi.cdef("""
typedef struct {
    void* data;
    GoInt len;
    GoInt cap;
} GoSlice;

typedef struct {
    const char *data;
    GoInt len;
} GoString;

GoInt Add(GoInt a, GoInt b);
double Cosine(double v);
void Sort(GoSlice values);
GoInt Log(GoString str);
""")

lib = ffi.dlopen("./libs/disease.so")

print("disease.Add(12,99) = %d" % lib.Add(12,99))
print("disease.Cosine(1) = %f" % lib.Cosine(1))

data = ffi.new("GoInt[]", [74,4,122,9,12])
nums = ffi.new("GoSlice*", {'data':data, 'len':5, 'cap':5})
lib.Sort(nums[0])
print("disease.Sort(74,4,122,9,12) = %s" % [
    ffi.cast("GoInt*", nums.data)[i] 
    for i in range(nums.len)])

data = ffi.new("char[]", b"Hello Bhojpur Disease!")
msg = ffi.new("GoString*", {'data':data, 'len':13})
print("log id %d" % lib.Log(msg[0]))