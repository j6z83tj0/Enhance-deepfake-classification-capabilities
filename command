python train.py --name ADM_resnet50 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/ADM/imagenet_ai_0508_adm --niter 100 --gpu_ids 1 --num_threads 12
python train.py --name VQDM_resnet50 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/VQDM/imagenet_ai_0419_vqdm --niter 100 --gpu_ids 1 --num_threads 12
python train.py --name Wukong_resnet50 --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/wukong/imagenet_ai_0424_wukong --niter 100 --gpu_ids 2 --num_threads 12


python eval.py --no_crop --batch_size 1 --eval_mode --threshold 0.01 --filter CONTOUR


MIX_version

python train.py --name MIX_ALL_no_aug_CvT_v2 --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/ --niter 100 --gpu_ids 1 --num_threads 12 --classes ADM,BigGAN,glide,Midjourney,stable_diffusion_v_1_4,stable_diffusion_v_1_5,VQDM,wukong
python train.py --name ADM_CVit --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/ADM/ --niter 100 --gpu_ids 0 --num_threads 12 

python train.py --name Midjourney_resnet50_GAUSSIAN_BLUR_RGB --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/Midjourney --niter 100 --gpu_ids 2 --num_threads 12 --filter ModeFilter

"CONTOUR":ImageFilter.CONTOUR,
"DETAIL":ImageFilter.DETAIL,
"EDGE_ENHANCE":ImageFilter.EDGE_ENHANCE,
"EMBOSS":ImageFilter.EMBOSS,
"FIND_EDGES":ImageFilter.FIND_EDGES,
"SMOOTH":ImageFilter.SMOOTH,
"SHARPEN":ImageFilter.SHARPEN,
"UnsharpMask":ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3),
"ModeFilter":ImageFilter.ModeFilter(5)
"GAUSSIAN_BLUR"}

python train.py --name MIX_ADM_MIDJOURNEY_ModeFilter_resnet50_RGB --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/ --niter 100 --gpu_ids 0 --num_threads 12 --classes ADM,Midjourney --filter ModeFilter
python train.py --name Midjourney_resnet50_GAUSSIAN_BLUR_RGB --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/Midjourney --niter 100 --gpu_ids 1 --num_threads 12


python train.py --name sdv14_fusingmodel_ycc_hsv_GAUSSIAN_BLUR --dataroot /home/EA301B/AI_generated_classfication/Dataset/GenImage/stable_diffusion_v_1_4 --niter 100 --gpu_ids 1 --num_threads 12 --filter GAUSSIAN_BLUR