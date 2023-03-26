Making inference Dataset for ACGPN virtual fitting 
## make_inference_dataset.ipynb 
This code makes datasets for ACGPN VITON SIZE(192 * 256) 
*Before start, this code needs detectron2 code and uploaded on it*

if you input image, it gives you lots of images for example (resized = 192 * 256)
- test_img_resized![image](https://user-images.githubusercontent.com/45056638/227788178-1e0d4aac-fb82-4858-b99a-aa0956baf3ef.png)

- test_pose_resized(human pose estimation .json file)

- test_label_resized(human segmentation)![image](https://user-images.githubusercontent.com/45056638/227788257-690c7cd8-4e08-4f53-90c3-78e09f920d59.png)

- test_color_resized(human with no bg)![image](https://user-images.githubusercontent.com/45056638/227788209-e9de329e-3501-479e-9bd0-41b00af03167.png)

- test_color_bg_resized(no human with bg)![image](https://user-images.githubusercontent.com/45056638/227788245-40c3af96-a5ee-42b1-870f-c4b88f5e04c4.png)

- test_pose_resized_img(human pose estimation img)![image](https://user-images.githubusercontent.com/45056638/227788284-b5654db4-fbed-4b40-a4af-deb9e1502d5c.png)

- test_edge_resized(human white bg black)![image](https://user-images.githubusercontent.com/45056638/227788219-e0410127-1696-4e7d-aae6-59fdc766ee71.png)



## u2net_human_seg_TDWV.py
This code makes test_edge image
![image](https://user-images.githubusercontent.com/45056638/227788160-99bf86d5-fd30-4bc4-8152-ba0198cadd05.png)


## ref
https://github.com/switchablenorms/DeepFashion_Try_On
https://github.com/facebookresearch/detectron2
https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN
