import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
from tqdm import tqdm


def validate(model, opt,val_data_loader= None):
    if val_data_loader != None :
        data_loader=val_data_loader
    else:
        data_loader,_ = create_dataloader(opt)
    print(opt.threshold)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img_rgb, img_hsv, label in tqdm(data_loader):
            img_rgb_tens = img_rgb.cuda()
            img_hsv_tens = img_hsv.cuda()
            y_pred.extend(model(img_rgb_tens,img_hsv_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > opt.threshold)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > opt.threshold)
    acc = accuracy_score(y_true, y_pred > opt.threshold)
    ap = average_precision_score(y_true, y_pred)
    real_list = [y_pred[i] for i, value in enumerate(y_true) if value == 0]
    fake_list = [y_pred[i] for i, value in enumerate(y_true) if value == 1]
    real_averscore=sum(real_list)/len(real_list)
    fake_averscore=sum(fake_list)/len(fake_list)

    return acc, ap, r_acc, f_acc, y_true, y_pred, real_averscore, fake_averscore


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
