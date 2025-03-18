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

    def compare_rtstruct_with_prediction(self, rtstruct_mask, predicted_mask, organ_idx, plot=True):
        organ_mask = (predicted_mask == organ_idx).astype(np.int32)
        
        intersection = np.logical_and(rtstruct_mask, organ_mask).sum()
        union = np.logical_or(rtstruct_mask, organ_mask).sum()
        dice = (2.0 * intersection) / (rtstruct_mask.sum() + organ_mask.sum() + 1e-6)
        jaccard = intersection / (union + 1e-6)
        
        metrics = {
            "dice": dice,
            "jaccard": jaccard,
            "rtstruct_volume_voxels": rtstruct_mask.sum(),
            "predicted_volume_voxels": organ_mask.sum(),
            "intersection_voxels": intersection,
            "union_voxels": union,
        }
        
        if plot:
            rtstruct_slices = np.where(rtstruct_mask.sum(axis=(0, 1)) > 0)[0]
            if len(rtstruct_slices) > 0:
                slice_idx = rtstruct_slices[len(rtstruct_slices) // 2]
                self.plot_comparison(rtstruct_mask, organ_mask, slice_idx, metrics)
        
        return metrics
    
    def plot_comparison(self, rtstruct_mask, predicted_mask, slice_idx, metrics=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rtstruct_mask[:, :, slice_idx].T, cmap='Blues')
        axes[0].set_title('RTSTRUCT Mask')
        axes[0].axis('off')
        
        axes[1].imshow(predicted_mask[:, :, slice_idx].T, cmap='Reds')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Create overlay with blue for RTSTRUCT and red for predictions
        overlay = np.zeros((*rtstruct_mask.shape[:2], 3))
        overlay[..., 0] = predicted_mask[:, :, slice_idx].T
        overlay[..., 2] = rtstruct_mask[:, :, slice_idx].T
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Blue: RTSTRUCT, Red: Predicted)')
        axes[2].axis('off')
        
        if metrics:
            fig.suptitle(f"Dice: {metrics['dice']:.4f}, Jaccard: {metrics['jaccard']:.4f}")
        
        plt.tight_layout()
        plt.show()
    
    def calculate_organ_volume(self, segmentation, organ_idx, voxel_spacing=None):
        num_voxels = (segmentation == organ_idx).sum().item()
        
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
            logger.warning("Using default voxel spacing of 1mm³.")
        
        voxel_volume_cm3 = np.prod(np.array(voxel_spacing)) / 1000.0
        organ_volume = num_voxels * voxel_volume_cm3
        
        logger.info(f"Organ volume: {organ_volume:.2f} cm³")
        return organ_volume
    
    def plot_segmentation(self, ct_image, segmentation, slice_idx, view="axial"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        if view == "axial":
            ct_slice = ct_image[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]
        elif view == "coronal":
            ct_slice = ct_image[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :]
        elif view == "sagittal":
            ct_slice = ct_image[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        else:
            raise ValueError(f"Invalid view: {view}")
        
        axes[0].imshow(ct_slice.T, cmap='Greys_r')
        axes[0].set_title(f'CT ({view} view)')
        axes[0].axis('off')
        
        axes[1].imshow(seg_slice.T, cmap='nipy_spectral')
        axes[1].set_title(f'Segmentation ({view} view)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_segmentation_as_rtstruct(self, segmentation, dicom_series_path, output_path=None):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(dicom_series_path), "segmentation.dcm")
        
        try:
            # Create a new RTStruct and add each segmented organ
            rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
            
            for organ_idx, organ_name in self.organ_indices.items():
                mask = (segmentation == organ_idx).astype(np.uint8)
                
                if mask.sum() == 0:
                    continue
                
                logger.info(f"Adding ROI for {organ_name}")
                rtstruct.add_roi(mask=mask, name=organ_name)
            
            rtstruct.save(output_path)
            logger.info(f"Saved RTSTRUCT to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving segmentation as RTSTRUCT: {e}")
            raise


# TO DO: EXAMPLE USAGE 