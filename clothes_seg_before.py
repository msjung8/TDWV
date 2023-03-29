#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pwd')


# In[2]:


import os
if os.path.isdir('./cloth-segmentation'):
    print("cloth-segmentation dir exists")
else:
    get_ipython().system('git clone https://github.com/levindabhi/cloth-segmentation')
    get_ipython().run_line_magic('cd', 'cloth-segmentation')
    get_ipython().system('pip install gdown')
    if not os.path.isfile('cloth_segm_u2net_latest.pth'):
        get_ipython().system('gdown --id 1mhF3yqd7R-Uje092eypktNl-RoZNuiCJ')
    get_ipython().system('mkdir input_images')
    get_ipython().system('mkdir output_images')
    get_ipython().run_line_magic('cd', '../')

