Making inference Dataset for ACGPN virtual fitting 
# make_inference_dataset.ipynb 
This code makes datasets for ACGPN VITON SIZE(192 * 256) 
if you input image, it gives you lots of images for example (resized = 192 * 256)
- test_img_resized
- test_pose_resized(human pose estimation .json file)
- test_label_resized(human segmentation)
- test_color_resized(human with no bg)
- test_color_bg_resized(no human with bg)
- test_pose_resized_img(human pose estimation img)
- test_edge_resized(human white bg black)
u2net_human_seg_TDWV.py
