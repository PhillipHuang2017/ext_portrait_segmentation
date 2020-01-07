#!/bin/bash
MODEL=SINet
#MODEL=ExtremeC3Net
VIDEO=./videos/ygj_car_0523.mp4
#VIDEO=./videos/cy_car_0523.mp4

CUDA_VISIBLE_DEVICES=1 python3 -m etc.Visualize_video \
--config=./setting/Test_${MODEL}.json \
--outputdir=./segVideos/ \
--weight=./result/${MODEL}/${MODEL}.pth \
--video=${VIDEO} \
--input_size=224
