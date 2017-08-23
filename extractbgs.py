#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Extract background images from a tar archive.

"""


__all__ = (
    'extract_backgrounds',
)


import os
import sys
import tarfile

import cv2
import numpy

from urllib.request import urlopen


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


def extract_backgrounds(archive_name, limit=None):
    """
    Extract backgrounds from provided tar archive.

    JPEGs from the archive are converted into grayscale, and cropped/resized to
    256x256, and saved in ./bgs/.

    :param archive_name:
        Name of the .tar file containing JPEGs of background images.

    """
    os.mkdir("bgs")

    if archive_name.startswith('http'):
        r = urlopen(archive_name)
        t = tarfile.open(fileobj=r, mode='r|*')
    else:
        t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f = t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue

        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))
        fname = "bgs/{:08}.jpg".format(index)
        if index % 100 == 0:
            print(fname)
        rc = cv2.imwrite(fname, im)
        if not rc:
            raise Exception("Failed to write file {}".format(fname))
        index += 1

        if limit and index >= limit:
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_name", default=None, help="")
    parser.add_argument("--limit", type=int, default=None, help="")
    args = parser.parse_args()

    archive_name = args.archive_name
    limit = args.limit

    extract_backgrounds(archive_name=archive_name, limit=limit)
