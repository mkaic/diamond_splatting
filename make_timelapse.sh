#!/bin/bash
# Create a high-quality MP4 timelapse from PNG sequence in ./outputs.
# Assumes files are named as 00000.png, 00001.png, etc.

ffmpeg -framerate 30 -i ./outputs/%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p timelapse.mp4