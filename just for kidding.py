# import torchvision.models as models
# import torch
# from torchsummary import summary
# model = models.vit_b_16()

# for name,_ in model.named_children():
#     print(name)
# from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
# import numpy as np

# a=np.array([1,1,1,1,0,1])
# b=np.array([0.1,0.2,0.3,0.8,0.6,0.8])
# print(a[a==0])
# print(b>0.5)
# print(accuracy_score(a,b>0.5))

# print(y_true==1)
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# with open("results/Midjourney_resnet50_SHARPEN_genimage_withfilter_SHARPEN_ADM.pkl","rb")as f:
#     data=pickle.load(f)
# y_true, y_pred= data
# y_pred_real= y_pred[y_true==0]
# y_pred_fake= y_pred[y_true==1]

# plt.bar(range(len(y_pred_real)), y_pred_real)

# plt.show()
# from networks.resnet import resnet50
# from torchinfo import summary

# model = resnet50()
# print(list(model.children())[:-2])
import torch
import torch.nn as nn
a=   torch.tensor([1,2,3])
b=   torch.tensor([4,5,6])

c= torch.add(a,b)
print(c)