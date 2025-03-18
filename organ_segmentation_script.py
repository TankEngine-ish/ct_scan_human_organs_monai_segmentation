import os
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from tcia_utils import nbia
from monai.bundle import ConfigParser, download
from monai.transforms import (
    LoadImage, LoadImaged, Orientation, Orientationd, 
    EnsureChannelFirst, EnsureChannelFirstd, Compose,
    Spacingd, ScaleIntensityRanged, CropForegroundd
)
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements
import json
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganSegmenter:
    def __init__(self, data_dir, model_name="wholeBody_ct_segmentation"):
        self.data_dir = data_dir
        self.model_name = model_name
        
        os.makedirs(data_dir, exist_ok=True)
        
        self.download_model()
        
        self.model_dir = os.path.join(data_dir, model_name)
        self.model_path = os.path.join(self.model_dir, 'models', 'model_lowres.pt')
        self.config_path = os.path.join(self.model_dir, 'configs', 'inference.json')
        
        self.config = ConfigParser()
        self.config.read_config(self.config_path)
        
        self.preprocessing = self._get_preprocessing_pipeline()
        self.model = self._load_model()
        self.inferer = self.config.get_parsed_content("inferer")
        
        self._modify_postprocessing_config()
        self.postprocessing = self.config.get_parsed_content("postprocessing")
        
        # Mapping of organ indices to names
        self.organ_indices = {
            1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder",
            5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
            9: "inferior_vena_cava", 10: "portal_vein_and_splenic_vein",
            11: "pancreas", 12: "right_adrenal_gland", 13: "left_adrenal_gland",
            14: "duodenum", 15: "bladder", 16: "prostate/uterus"
        }

    def download_model(self):
        if not os.path.exists(os.path.join(self.data_dir, self.model_name)):
            logger.info(f"Downloading {self.model_name}...")
            download(name=self.model_name, bundle_dir=self.data_dir)
            logger.info("Download complete.")
        else:
            logger.info(f"Model {self.model_name} already exists.")
    
    def _get_preprocessing_pipeline(self):
        # Create enhanced preprocessing pipeline for handling variable inputs
        enhanced_preprocessing = Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="LPS"),
            Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ScaleIntensityRanged(keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True),
            CropForegroundd(keys="image", source_key="image", select_fn=lambda x: x > 0.05),
        ])
        
        return enhanced_preprocessing
    
    def _modify_postprocessing_config(self):
        try:
            postprocessing_list = self.config["postprocessing"]
            modified_list = [item for item in postprocessing_list 
                            if "SaveImaged" not in str(item)]
            self.config["postprocessing"] = modified_list
            logger.info("Removed SaveImaged from postprocessing pipeline.")
        except Exception as e:
            logger.warning(f"Could not modify postprocessing config: {e}")
    
    def _load_model(self):
        model = self.config.get_parsed_content("network")
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def segment_from_folder(self, ct_folder):
        logger.info(f"Processing CT folder: {ct_folder}")
        
        try:
            data = self.preprocessing({'image': ct_folder})
            
            with torch.no_grad():
                data['pred'] = self.inferer(data['image'].unsqueeze(0), network=self.model)
            
            data['pred'] = data['pred'][0]
            data['image'] = data['image'][0]
            
            data = self.postprocessing(data)
            
            segmentation = data['pred'][0]
            segmentation = torch.flip(segmentation, dims=[2])
            segmentation = segmentation.cpu().numpy()
            
            return data['image'][0].cpu().numpy(), segmentation
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            raise


























































#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import numpy as np
import torch
import pydicom
import matplotlib.pyplot as plt
from tcia_utils import nbia
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, LoadImaged, Orientation, Orientationd, EnsureChannelFirst, EnsureChannelFirstd, Compose
from rt_utils import RTStructBuilder
from scipy.ndimage import label, measurements
import json


# In[ ]:


datadir = '/home/nito/Projects/ct_scan_human_organs_monai_segmentation'


# # Part 1: Open CT Image

# First we'll download the CT data we'll be using to the data directory
# 
# * This data was obtained from the cancer imaging archive

# In[11]:


cart_name = "nbia-74551708346483681"
cart_data = nbia.getSharedCart(cart_name)
df = nbia.downloadSeries(cart_data, format="df", path = datadir)


# In[12]:


CT_folder = os.path.join(datadir, '1.3.6.1.4.1.14519.5.2.1.3320.3273.193828570195012288011029757668')


# ## Option 1: Using `pydicom`

# In[24]:


ds = pydicom.read_file(os.path.join(CT_folder, '1-669.dcm'))


# In[25]:


ds


# We can obtain pixel data by accessing the `pixel_array` attribute

# In[26]:


image = ds.pixel_array
image.shape


# Note that the image is a 2D array. Typically the pixel values are stored in a scaled format so we should adjust them back to Hounsifled units (Hounsfield units (HU) are a dimensionless unit universally used in computed tomography (CT) scanning to express CT numbers in a standardized and convenient form)

# In[27]:


image = ds.RescaleSlope * image + ds.RescaleIntercept


# 

# In[28]:


plt.pcolormesh(image, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()


# ## Option 2: Using `monai`

# MONAI stands for "Medical Open Network for Artificial Intelligence" and is essentially an extension of PyTorch for machine learning with medical data, containing **many many many** important functions. If you're doing AI research in medicine, you **must** use MONAI.
# 
# MONAI has functionality for easily opening up medical data:

# In[29]:


image_loader = LoadImage(image_only=True)
CT = image_loader(CT_folder)


# In[30]:


CT


# The CT contains both the pixel data (for all slices) and the image metadata

# In[31]:


CT.meta


# Now we can plot any plane of the CT image we like

# In[36]:


CT_coronal_slice = CT[:,256].cpu().numpy()


# View CT image

# In[37]:


plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()


# Notice that he's upside-down! We can manually reverse the axis, or we can use MONAI functionality to modify the CT. Firstly we add a channel dimension, since this is required for most AI applications

# In[38]:


CT.shape


# In[39]:


channel_transform = EnsureChannelFirst()
CT = channel_transform(CT)
CT.shape


# Now we can reorient the CT image. LPS corresponds to L = left, P = posterior, S = superior.

# In[43]:


orientation_transform = Orientation(axcodes=('LPS'))
CT = orientation_transform(CT)


# Now obtain the coronal slice

# In[44]:


CT_coronal_slice = CT[0,:,256].cpu().numpy()


# Now plot again

# In[45]:


plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()


# Alternatively, we can combine all these transforms in one go when we open the image data:

# In[46]:


preprocessing_pipeline = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes='LPS')
])


# And we can open using this preprocessing pipeline:

# In[47]:


CT = preprocessing_pipeline(CT_folder)
CT_coronal_slice = CT[0,:,256].cpu().numpy()


# And plot:

# In[48]:


plt.figure(figsize=(3,8))
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.colorbar(label='HU')
plt.axis('off')
plt.show()


# One other option (which is typically done) is to use the "dictionary" version of all the transforms above. This is done by adding a `d` to the end of the transforms, such as `LoadImaged`.
# * These transforms take in a dictionary with keys-value pairs

# In[49]:


data = {'image': CT_folder, 'some_other_key': 42}


# In[52]:


preprocessing_pipeline = Compose([
    LoadImaged(keys='image', image_only=True),
    EnsureChannelFirstd(keys='image'),
    Orientationd(keys='image',axcodes='LPS')
])


# In[53]:


data = preprocessing_pipeline(data)


# In[54]:


data


# # Part 2: Segmentation Model

# First we'll download the segmentation model
# 
# * Obtained from https://monai.io/model-zoo.html NB: This model is no longer available for download! MUST BE REPALCED WITH ANOTHER!
# 

# In[ ]:


model_name = "wholeBody_ct_segmentation"
download(name=model_name, bundle_dir=datadir)


# We first set the paths of where we downloaded the model parameters (`model.pt`) and a file called `inference.json`.

# In[82]:


model_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
config_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'configs', 'inference.json')


# From this we create a `config` instance which lets us read from the `json` file

# In[83]:


config = ConfigParser()
config.read_config(config_path)


# ## Preprocessing Pipeline

# From this we can extract the preprocessing pipeline specified by the `inference.json` file
# * These are all the operations applied to the data before feeding it to the model

# In[68]:


preprocessing = config.get_parsed_content("preprocessing")


# Note that this preprocessing pipeline uses `LoadImaged` instead of `LoadImage`. The `d` at the end refers to the fact that everything should be fed in as a dictionary. The `keys` argument are the keys of the dictionary by which to apply the transform to

# In[69]:


data = preprocessing({'image': CT_folder})


# In this case, the operations have only been applied to things with the key `'image'`. We could add extra keys and nothing would happen.

# ## Model

# Now we can obtain the model using the `'network'` key from the json file

# In[70]:


model = config.get_parsed_content("network")


# Note at the moment that the model is initialized with random parameters. We need to configure it with the parameters given by the `model.pt` file. Since we won't be training it (only use it for evaluation), we'll use the `eval()` function

# In[71]:


model.load_state_dict(torch.load(model_path))
model.eval();


# ## Inferer

# The `"inferer"` pipeline takes in the data and the model, and returns model output. It contains some extra processing steps (in this case it breaks the data into 96x96x96 chunks before feeding it into the model)

# In[72]:


inferer = config.get_parsed_content("inferer")


# ## Postprocessing

# Finally, once the model has finished running, there will be postprocessing that needs to be done on the data
# 
# * **IMPORTANT**: The postprocessing automaticcally saves the data to disk. I have manually deleted the `"SaveImaged"` from the postprocessing pipeline in the json file

# In[84]:


postprocessing = config.get_parsed_content("postprocessing")


# In[ ]:


data['image'].unsqueeze(0).shape


# # Prediction Time :)

# We can now combine all these pipelines to obtain organ masks for our data

# In[85]:


data = preprocessing({'image': CT_folder}) # returns a dictionary
# 2. Compute mask prediction, add it to dictionary
with torch.no_grad():
    # Have to add additional batch dimension to feed into model
    data['pred'] = inferer(data['image'].unsqueeze(0), network=model)
# Remove batch dimension in image and prediction
data['pred'] = data['pred'][0]
data['image'] = data['image'][0]
# Apply postprocessing to data
data = postprocessing(data)
segmentation = torch.flip(data['pred'][0], dims=[2])
segmentation = segmentation.cpu().numpy()


# In[35]:


slice_idx = 250
CT_coronal_slice = CT[0,:,slice_idx].cpu().numpy()
segmentation_coronal_slice = segmentation[:,slice_idx]


# In[37]:


plt.subplots(1,2,figsize=(6,8))
plt.subplot(121)
plt.pcolormesh(CT_coronal_slice.T, cmap='Greys_r')
plt.axis('off')
plt.subplot(122)
plt.pcolormesh(segmentation_coronal_slice.T, cmap='nipy_spectral')
plt.axis('off')
plt.show()


# **Example**: Computing bladder volume

# In[39]:


number_bladder_voxels = (segmentation==13).sum().item()
voxel_volume_cm3 = np.prod(CT.meta['spacing']/10)
bladder_volume = number_bladder_voxels * voxel_volume_cm3
print(f'Bladder Volume {bladder_volume:.1f}cm^3')

