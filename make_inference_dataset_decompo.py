#!/usr/bin/env python
# coding: utf-8

# # 순서
# ### import modules
# ### methods
# ### predictor 초기화
# ### 전처리

# In[1]:


import os


# In[2]:


get_ipython().system('pwd')


# In[3]:


# U_2_Net 설치
existdir = './U_2_Net'
if os.path.isdir(existdir):
    print('U_2_Net exist')
else:
#get_ipython().system('if [! -d ./U_2_Net]; then')
#s.makedirs(./U_2_Net, exist_ok=True)
    get_ipython().system('mkdir U_2_Net')
    get_ipython().system('git clone https://github.com/NathanUA/U-2-Net.git U_2_Net')
    get_ipython().run_line_magic('cd', 'U_2_Net')
    get_ipython().system('git clone https://github.com/NKAnzu/TDWV.git')
    get_ipython().system('mv TDWV/u2net_human_seg_TDWV.py ./')
    get_ipython().run_line_magic('cd', 'U_2_Net/saved_models/')
    get_ipython().system('mkdir u2net_human_seg')
    get_ipython().run_line_magic('cd', 'u2net_human_seg')
    existfile='u2net_human_seg.pth'
    if not os.path.isfile(existfile):
        get_ipython().system('gdown https://drive.google.com/uc?id=1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P')
    get_ipython().run_line_magic('cd', '../../')
    


# In[4]:


get_ipython().run_line_magic('cd', 'U_2_Net')


# In[5]:


import U_2_Net.u2net_human_seg_TDWV
import U_2_Net.data_loader


# In[6]:


get_ipython().run_line_magic('cd', '../')


# In[7]:


get_ipython().system('pip install imutils')
get_ipython().system('pip install pycocotools')


# # human segmentation (test_label)

# In[8]:


pip install ninja


# In[9]:


pip install gdown


# # mkdir Dataset 
# 

# In[10]:


get_ipython().system('pwd')
if os.path.isdir('./datasets'):
    print('./datasets exist')
else:
    get_ipython().system('mkdir datasets')
    get_ipython().run_line_magic('cd', 'datasets')
    get_ipython().system('mkdir new_datasets')
    get_ipython().run_line_magic('cd', 'new_datasets')
    get_ipython().system('mkdir test_img')
    get_ipython().system('mkdir test_img_resized')
    
    get_ipython().system('mkdir test_pose')
    get_ipython().system('mkdir test_pose_resized')
    get_ipython().system('mkdir test_pose_resized_img')
    
    get_ipython().system('mkdir test_label')
    get_ipython().system('mkdir test_label_resized')
    
    get_ipython().system('mkdir test_edge')
    get_ipython().system('mkdir test_edge_resized')
    
    get_ipython().system('mkdir test_color')
    get_ipython().system('mkdir test_color_resized')
    get_ipython().system('mkdir test_color_bg_resized')
    
    get_ipython().system('mkdir test_clothes')
    get_ipython().system('mkdir test_clothes_upper')
    get_ipython().system('mkdir test_clothes_upper_resized')
    
    get_ipython().system('mkdir test_clothes_lower')
    get_ipython().system('mkdir test_clothes_lower_resized')
    get_ipython().run_line_magic('cd', '../../')


# # 이 이후로 이미지 생성처리가 시작됨

# https://github.com/levindabhi/cloth-segmentation git clone> clothes segmentation

# In[11]:


get_ipython().system('pwd')


# In[12]:


get_ipython().run_line_magic('cd', 'datasets/new_datasets/test_img/')


# In[13]:


get_ipython().system('rm -rf .ipynb_checkpoints')


# In[14]:


get_ipython().run_line_magic('cd', '../../../')


# ## clothes_seg.py input result dir 변경하고 하기

# In[15]:


get_ipython().system('pwd')


# In[17]:


get_ipython().system('rm -rf TDWV')
if not os.path.isdir('TDWV'):
    get_ipython().system('git clone https://github.com/NKAnzu/TDWV.git')
    get_ipython().system('cp TDWV/clothes_seg.py ./clothes_seg.py')
    get_ipython().system('cp TDWV/clothes_seg_before.py ./clothes_seg_before.py')
if not os.path.isfile('clothes_seg_before.py'):
    get_ipython().system('cp TDWV/clothes_seg.py ./clothes_seg.py')
    get_ipython().system('cp TDWV/clothes_seg_before.py ./clothes_seg_before.py')
elif not os.path.isfile('clothes_seg.py'):
    get_ipython().system('cp TDWV/clothes_seg.py ./clothes_seg.py')
    get_ipython().system('cp TDWV/clothes_seg_before.py ./clothes_seg_before.py')
    


# In[18]:


get_ipython().system('ipython clothes_seg_before.py')


# !ipython clothes_seg.py

# # 경로 설정

# ## test_img에 이미지 최소 1개 업로드 후 시작

# In[19]:


get_ipython().run_line_magic('cd', 'U_2_Net')


# In[20]:


import u2net_human_seg_TDWV as human_seg
#u2net_human_seg_TDWV.normPRED


# In[21]:


get_ipython().run_line_magic('cd', '../')


# In[22]:


if os.path.isdir('./Self-Correction-Human-Parsing-for-ACGPN'):
    print("Self-Correction-Human-Parsing-for-ACGPN dir exist")
else:
    get_ipython().system('git clone https://github.com/levindabhi/Self-Correction-Human-Parsing-for-ACGPN')
    get_ipython().run_line_magic('cd', 'Self-Correction-Human-Parsing-for-ACGPN')
    get_ipython().system('mkdir checkpoints')
    get_ipython().system('mkdir inputs')
    get_ipython().system('mkdir outputs')
    get_ipython().system('git clone https://github.com/NKAnzu/TDWV.git')
    get_ipython().system('mv TDWV/simple_extractor2.py ./')
    get_ipython().system('mv TDWV/simple_extractor_dataset.py ./datasets')
    
    get_ipython().run_line_magic('cd', 'checkpoints')
    #simple extractor2 , datasets/simpleextracdataset  git clone
    
    if not os.path.isfile('exp-schp-201908261155-lip.pth'):
        get_ipython().system('gdown https://drive.google.com/uc?id=1X6ytlE2itAPpxHX3xDxi7fuRM-7mUxM6')
    get_ipython().run_line_magic('cd', '../../')


# In[23]:


get_ipython().system('pwd')

