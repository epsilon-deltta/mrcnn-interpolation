### What about?  Instance segmentation model : Mask-RCNN !   
### So what : Improved performance!

> input
![](./sample/pds1.jpg)

**Mask-RCNN Caculates....**

> output
![](./sample/pre_pds1.jpg)


### Mask rcnn transfrom layer Interpolation experiments

Before Image goes into mask rcnn backbone module , image is resized to fixed size to get good features. 
When resizing image, generally Bilinear interpolation is used.
There're many kind of interpolation methods to resize in this repository code. 

![](./assets/main.jpg)
![](./assets/maskap.jpg)


## Train
### main options
- --dt : dataset

    - pf  : pennFudan
    - bln : balloon
    
- --model  : interpolation method

    - bicubic (👍)
    - bilinear
    - nearest



> Usage

```{.bash}  
python train.py --dt pf --model  bicubic -o ./model/something.pth
```



### Secondary options 

- --out     : default = './model/new.pth'

- --epochs  : default = 50
- --batch   : default = 4
- --device  : default = 'cuda:0'
- --workers : default = 4 

<!-- ### pretrained model 
(not read) We will provide this pretrained models soon.  
😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢😢       -->

## Evaluate
<!-- About model itself  -->
<!-- - the number of parameter  -->
About performance
- mAP
    - mask
    - bbox

> Jupyter (interactive)  

: evaluate.ipynb 

> Shell

```bash:howtoevaluate
python evaluate.py -m /path/to/modelA.pth /path/to/modelB.pth -o /where/to/save/figure_dir
```
### options
- -m ,--model (default) ./models/*.pth 
- -o ,--output (default) false (false : prints evaluation results on console, true : saves graph images in ./results directory )

e.g )
```bash:howtoevaluate
python evaluate.py 
python evaluate.py -m /path/to/modelA.pth /path/to/modelB.pth -o true
python evaluate.py -o true
python evaluate.py -m /path/to/modelA.pth /path/to/modelB.pth 
```

image output : modelName_ap_epochs.jpg , modelName_ap_table.jpg

## Inference
Adjust bicubic mask rcnn to your image.

> Bash Usage

```bash:  
python inference.py -m ./models/pf_4_nearest.pth -i ./input.jpg -o ./output.jpg
```
### options
- -m ,--model (default) './models/pf_4_bicubic.pth' 
- -i ,--input (default) './sample/pds1.jpg'
- -o ,--output (default) './sample/pre_pds1.jpg' 

<!-- > Library Usage -->

## ETC

  If you have a question , feel free to ask me.

<!-- > **reference** -->

