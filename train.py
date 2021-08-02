import argparse
import importlib
import torch
import utils

if __name__ == '__main__' :
    exam_code = '''
    e.g)  
    python train.py --dt pf --model pf_bicubic
    '''
    parser = argparse.ArgumentParser("Train Mask R-CNN model",epilog=exam_code)   
    # setting
    parser.add_argument('-d'  ,'--dt'      ,default='pf'      ,metavar='{pf,bln}' , help='Dataset')
    parser.add_argument('-m'  ,'--model'   ,default='bicubic' ,metavar='{...}'    ,help='model class name')
    parser.add_argument('-pre','--pretrain',default='none'    ,metavar='{...}'    ,help='pretrained model file path')
    parser.add_argument('-o'  ,'--out'    ,default= './model/new.pth'             ,help='model path to save')
    #hyper param
    parser.add_argument('-epo','--epochs'    ,default= 50        ,type = int         ,help='number of epochs')
    parser.add_argument('-b'  ,'--batch'     ,default= 4         ,type = int         ,help='batch size')
    parser.add_argument('-w'  ,'--workers'   ,default= 4         ,type = int         ,help='number of batch workers')
    parser.add_argument('-de'  ,'--device'   ,default= 'cuda:0'                      ,help='device e.g} cuda:0')

    args = parser.parse_args()

    # args edit
    args.dt = args.dt.lower()
    if args.dt == 'pf' or args.dt == 'pennfudan':
        args.dt   = 'PennFudan'
        num_classes   = 2 
    elif args.dt == 'bln':
        args.dt = 'balloon'
        num_classes   = 2 

    if args.out == './model/new.pth':
        args.out = f'./model/{args.dt}_{args.model}.pth'

    import pprint
    pprint.pprint(args)


    ###### changeable main 
    from datasets import get_dataset
    from models   import get_models
    dataset , dataset_test = get_dataset(args.dt)
    model                  = get_models(name = args.model,num_classes = num_classes) 
    ######

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=utils.collate_fn)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device = args.device

    

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
        

    num_epochs = args.epochs
    from engine import train_one_epoch , evaluate
    evaluators = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
    #     device = 'cuda:1'
    #     model.to(device)
        evaluators.append( evaluate(model, data_loader_test, device=device) )

    torch.save({'state_dict':model.state_dict(),
                'evaluators':evaluators
           },args.out)