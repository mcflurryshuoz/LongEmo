#!/usr/bin/env python3
"""Run the benchmark's frame extraction on Ubuntu's older FFmpeg 4.2.

FFmpeg 4.2 does not recognize the newer ``-fps_mode`` and
``-enc_time_base:v`` options used by the normal loader.  This wrapper removes
only those two options; it does not alter model input or output files.
"""
import os
import sys


def main():
    args = []
    skip = False
    for value in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if value == "-fps_mode":
            # FFmpeg 4.2 uses the older spelling for passthrough timestamps.
            skip = True
            args.extend(("-vsync", "0"))
            continue
        if value == "-enc_time_base:v":
            skip = True
            continue
        args.append(value)
    os.execv("/usr/bin/ffmpeg", ["ffmpeg", *args])


if __name__ == "__main__":
    main()
