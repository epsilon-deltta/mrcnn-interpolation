### Mask rcnn transfrom layer Interpolation experiments

Before Image goes into mask rcnn backbone module , image is resized to fixed size to get good features. 
When resizing image, generally Bilinear interpolation is used.
In this repository code, there're many kind of interpolation methods to resize 

![](./assets/main.jpg)

## train
### main options
- --dt : dataset

    - pf  : pennFudan
    - bln : balloon
    
- --model  : interpolation method

    - bicubic (👍)
    - bilinear
    - nearest


> Usage

'''{.bash}  
python train.py --dt pf --model  bicubic -o ./model/something.pth
'''


### Secondary options 

- --out     : default = './model/new.pth'

- --epochs  : default = 50
- --batch   : default = 4
- --device  : default = 'cuda:0'
- --workers : default = 4 

## evaluate

## inference


> **reference**
