#!/bin/bash
# Create a high-quality MP4 timelapse from jpg sequence in ./outputs.
# Assumes files are named as 00000.jpg, 00001.jpg, etc.

ffmpeg -framerate 24 -i ./outputs/%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p timelapse.mp4 -y