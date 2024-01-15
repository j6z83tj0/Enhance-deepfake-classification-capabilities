from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

csv_name='_genimage'

# root to the testsets
dataroot = '/home/EA301B/AI_generated_classfication/Dataset/GenImage_test_small'   #genimage


# list of synthesis algorithms
vals = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5',
        'VQDM', 'wukong']                                   #genimage

# indicates if corresponding testset has multiple classes
multiclass = [0, 0, 0, 0, 0, 0, 0, 0]                         #genimage


# model
model_path_list=['checkpoints/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE.pth']
# model_path_list = ['checkpoints/sdv14_fusingmodel_ycc_hsv/sdv14_fusingmodel_ycc_hsv.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_CONTOUR/sdv14_fusingmodel_ycc_hsv_CONTOUR.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_DETAIL/sdv14_fusingmodel_ycc_hsv_DETAIL.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE/sdv14_fusingmodel_ycc_hsv_EDGE_ENHANCE.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_EMBOSS/sdv14_fusingmodel_ycc_hsv_EMBOSS.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_FIND_EDGES/sdv14_fusingmodel_ycc_hsv_FIND_EDGES.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_SMOOTH/sdv14_fusingmodel_ycc_hsv_SMOOTH.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_SHARPEN/sdv14_fusingmodel_ycc_hsv_SHARPEN.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_UnsharpMask/sdv14_fusingmodel_ycc_hsv_UnsharpMask.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_ModeFilter/sdv14_fusingmodel_ycc_hsv_ModeFilter.pth',
#                    'checkpoints/sdv14_fusingmodel_ycc_hsv_GAUSSIAN_BLUR/sdv14_fusingmodel_ycc_hsv_GAUSSIAN_BLUR.pth']

