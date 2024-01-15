import os
import csv
import torch
import pickle

from validate import validate
from networks.resnet import resnet50, resnet18
from options.test_options import TestOptions
from eval_config import *
from tqdm import tqdm
from vit_pytorch.cvt import CvT
from vit_pytorch.cross_vit import CrossViT
from networks.fusing_resnet import fusing_resnet ,fusing_resnet_add
# If you want to test different filters
# filters=[None,"CONTOUR","DETAIL","EDGE_ENHANCE","EMBOSS","FIND_EDGES","SMOOTH","SHARPEN","UnsharpMask","ModeFilter","GAUSSIAN_BLUR"]  

# Running tests
opt = TestOptions().parse(print_options=False)
for i,model_path in enumerate(model_path_list) :
    model_name = os.path.basename(model_path).replace('.pth', '')
    # If you want to test different filters
    # opt.filter=filter[i]
     
    rows = [["{} model testing on...with:{}".format(model_name,opt.filter)],
            ['testset', 'accuracy', 'avg precision','real_accuracy','fake_accuracy','real_scsore','fake_score']] 

    print("{} model testing on...".format(model_name))
    for v_id, val in tqdm(enumerate(vals),total=len(vals)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
        opt.no_resize = False    # testing without resizing by default

        model = fusing_resnet()
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model.cuda()
        model.eval()

        acc, ap, r_acc, f_acc, y_true, y_pred,real_averscore, fake_averscore= validate(model, opt)
        rows.append([val, acc, ap, r_acc, f_acc,real_averscore,fake_averscore])
        print("({}) acc: {}; ap: {}; r_acc: {}; f_acc: {}".format(val, acc, ap,r_acc,f_acc))
        # addition
        # data=(y_true, y_pred)                                        
        # csv_name = results_dir + '/{}_withfilter_{}_{}.pkl'.format(model_name+csv_name,opt.filter,val)
        # with open(csv_name, 'wb') as f:
        #     pickle.dump(data,f)
        # csv_name='_genimage'
    threshold = f"{opt.threshold:0.2f}".replace(".", "")
    csv_name = results_dir + '/{}_withfilter_{}_{}.csv'.format(model_name+csv_name,opt.filter,threshold)
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)
    csv_name='_genimage'
