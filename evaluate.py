import argparse
import importlib
import torch
import utils

import os
import sys
def get_paths(models):

    fnames = []
    model_dir = './models/'

    if models == 'all' : # search all models in './models' directory
        fnames = [ os.path.join(model_dir,x) for x in os.listdir(model_dir) if os.path.splitext(x)[-1] == '.pth' ]
    elif all([os.path.sep not in name for name in models] ) :
        # e.g) modelA.pth modelB.pth modelC.pth ...
        for fname in models :
            mpath = os.path.join(model_dir,fname)
            if not os.path.exists(mpath):
                print(f"{mpath} doesn't exist in ./models/ directory")
                sys.exit(1)
            fnames.append(mpath)

    else : # e.g) /path/to/modelA.pth /path/to/modelB.pth ...
        for fname in models :
            if not os.path.exists(fname):
                print(f"{fname} doesn't exist in ./models/ directory")
                sys.exit(1)
            fnames.append(fname)
    return fnames
def get_fnames(paths):
    return [os.path.splitext(os.path.split(path)[-1])[0] for path in paths ]

if __name__ == '__main__' :
    exam_code = '''
    python evaluate.py -m /path/to/modelA.pth /path/to/modelB.pth -o /where/to/save/figure_dir
    '''
    desc   = 'Evaluate'
    parser = argparse.ArgumentParser(f"{desc} Mask R-CNN model",epilog=exam_code)
    # setting
    # -m ~.pth , all(all of models in ./models)
    # -o dir ,none (console)

    
    parser.add_argument('-m'  ,'--model',nargs='*'   ,default='all' ,metavar='{...}'    ,help='model names')
    
    parser.add_argument('-o'  ,'--out'  ,default= 'false'           ,help='(false)printing or (true)saving file')
    
    args = parser.parse_args()

    fpaths = get_paths(args.model)
    print(fpaths)
    fnames  = get_fnames(fpaths) #

    for i,path in enumerate(fpaths):
        exec(f'{fnames[i]}_pter  = torch.load("{path}")')
        try:
            exec(f"{fnames[i]}_evals = {fnames[i]}_pter['evaluators']")
        except Exception as e :
            exec(f"{fnames[i]}_evals = {fnames[i]}_pter['evaluator']")
            print(e)

    save_dir = './results/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    from matplotlib import pyplot as plt
    interpols     = fnames # e.g) ['pf_4_bicubic','pf_4_nearest','pf_4_bilinear']
    metric_name   = ['ap','ap50','ap75','Maskap','Maskap50','Maskap75']
    allname       = '&'.join(interpols)
    num_metric    = len(metric_name)

    # comparison figure
    # img ✔ print Ⅹ

    limit         = 10
    if args.out != 'false':
        f, axs = plt.subplots(num_metric, 1, figsize=(12, 20),sharex=False) # w,h
        for inter in interpols:
            exec(f'evaluators={inter}_evals')
            
            metrics = dict(zip( metric_name, list( [] for _ in range(len(metric_name))  )  
                            ))

            for evtor in evaluators:
                for iou_type, coco_eval in evtor.coco_eval.items():
                    #coco_eval.summarize()
                    #  iou_type : 'bbox' ,'segm'
                    if iou_type == 'bbox':
                        aps = coco_eval.stats[:3]
                        metrics['ap'].append(aps[0])
                        metrics['ap50'].append(aps[1])
                        metrics['ap75'].append(aps[2])

                    else : # 'segm'
                        aps = coco_eval.stats[:3]
                        metrics['Maskap'].append(aps[0])
                        metrics['Maskap50'].append(aps[1])
                        metrics['Maskap75'].append(aps[2])
            
            metrics_keys = list(metrics.keys() )
            for i,met_name in enumerate(metrics):
                axs[i].set_title(met_name)
                axs[i].plot( metrics[met_name][:limit],'-+',label=inter)
                axs[i].legend()
        plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=1)
 
        plt.savefig(os.path.join(save_dir,f'{allname}_ap_epochs.jpg') )

    # performance table (at certain epoch)
    # img ✔ print ✔
    import pandas as pd
    num_epoch = 11
    aps = []
    for inter in interpols:
        exec(f"bbox = {inter}_evals[num_epoch].coco_eval['bbox'].stats[:3]")
        exec(f"segm = {inter}_evals[num_epoch].coco_eval['segm'].stats[:3]")
        row = bbox.tolist()+segm.tolist()
        aps.append(row)

    df = pd.DataFrame(aps,index=interpols,columns=metric_name)
    
    if args.out != 'false':
        df.to_csv(os.path.join(save_dir,f'{allname}_ap_table.csv'))
    
    print(f'At {num_epoch} epoch \n')
    print(df)
    # command