#!/bin/sh

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip #axel doesn't work 
unzip -q PennFudanPed.zip

wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip -d balloon