from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, ToTensor, ToPILImage
import torch
import glob
from skimage.morphology import square
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple
from transforms.input import GaussianNoise, EnhanceContrast, EnhanceColor
from transforms.target import Opening, ConvexHull, BoundingBox
import numpy as np
import funcy
import matplotlib.pyplot as plt
import json

unloader = ToPILImage()
def imshow(tensor):
	image = unloader(tensor)
	plt.imshow(image)
	plt.pause(0.01)

class ISIC(Dataset):

	def __init__(self, img_dir: str, mask_dir: str,  augmentations: List =None, input_preprocess: Callable =None, target_preprocess: Callable =None,
				 with_targets: bool=True, size: Tuple =(256, 256), phase = 'training'):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.size = size
		self.load_superpixel_label = False
		if phase == 'training':
			self.load_superpixel_label = True
   
		self.normalize =Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
		self.ids = [file for file in os.listdir(self.img_dir)
					if not file.startswith('.')]
		self.ids.sort(key=lambda x: int(x.split('.')[0][5:]))
		self.resize = Resize(size=self.size)

		self.to_tensor = ToTensor()
		self.input_preprocess = input_preprocess
		self.target_preprocess = target_preprocess
		if augmentations:
			augmentations = [lambda x: x] + augmentations
		else:
			augmentations = [lambda x: x ]

		# self.data = [(idx, augmentation) for augmentation in augmentations for idx in self.ids]
	@staticmethod
	def _load_input_image(fpath: str):
		img = Image.open(fpath).convert("RGB")
		return img

	@staticmethod
	def _load_target_image(fpath: str):
		img = Image.open(fpath).convert("L")
		return img


	def __len__(self):
		# print(self.data)

		return len(self.ids) # 200 * 3

	def __getitem__(self, i):
		# idx, augmentation = self.data[i]
		# print(idx, augmentation)
		idx= self.ids[i]
		# augmentation = self.data[i]
		fullPathName = os.path.join(self.img_dir, idx)
		fullPathName = fullPathName.replace('\\', '/')
		img = self._load_input_image(fullPathName)
		img = self.resize(img)

		if self.input_preprocess is not None:
			img = self.input_preprocess(img)

		# img = augmentation(img)
		img = self.to_tensor(img)
		img = self.normalize(img)

		# get mask
		MaskPathName = os.path.join(self.mask_dir, idx)
		MaskPathName = MaskPathName.replace('\\', '/').split('.')[:-1]
		MaskPathName = '.'.join(MaskPathName)+'_segmentation.png'

		try:
			final_path = glob.glob(MaskPathName)
			Mask = self._load_target_image(final_path[0])
			Mask = self.resize(Mask)
			Mask = np.array(Mask)
			Mask = np.where(Mask != 0, 1, 0)

		except:
			Mask = np.zeros((256,256))

		if self.load_superpixel_label:
			superpixel_dir = '/'.join(self.mask_dir.split('/')[0:3] + ["superpixel_labels"])
			superpixelPathName = os.path.join(superpixel_dir, idx)
   
			superpixelPathName = superpixelPathName + '.json'
			with open(superpixelPathName, 'r') as f:
				superpixel = json.load(f)
				superpixel = np.array(superpixel)
		# if self.target_preprocess is not None:
		# 	Mask = self.target_preprocess(Mask)
		Mask = self.to_tensor(Mask)
		# print(Mask.size())
		if self.load_superpixel_label:
			superpixel =  self.to_tensor(superpixel)
			return {'image': img, 'mask': Mask,'superpixel':superpixel, 'name': idx}
		else:
			return {'image': img, 'mask': Mask, 'name': idx}

class PH2(Dataset):

	def __init__(self, img_dir: str, mask_dir: str,  augmentations: List =None, input_preprocess: Callable =None, target_preprocess: Callable =None,
				 with_targets: bool=True, size: Tuple =(256, 256)):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.size = size
		self.normalize =Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
		self.ids = [file for file in os.listdir(self.img_dir)
					if not file.startswith('.')]
		self.ids.sort(key=lambda x: int(x.split('.')[0][5:]))
		self.resize = Resize(size=self.size)

		self.to_tensor = ToTensor()
		self.input_preprocess = input_preprocess
		self.target_preprocess = target_preprocess
		if augmentations:
			augmentations = [lambda x: x] + augmentations
		else:
			augmentations = [lambda x: x ]

		# self.data = [(idx, augmentation) for augmentation in augmentations for idx in self.ids]
	@staticmethod
	def _load_input_image(fpath: str):
		img = Image.open(fpath).convert("RGB")
		return img

	@staticmethod
	def _load_target_image(fpath: str):
		img = Image.open(fpath).convert("L")
		return img


	def __len__(self):
		# print(self.data)

		return len(self.ids) # 200 * 3

	def __getitem__(self, i):
		# idx, augmentation = self.data[i]
		# print(idx, augmentation)
		idx= self.ids[i]
		# augmentation = self.data[i]
		fullPathName = os.path.join(self.img_dir, idx)
		fullPathName = fullPathName.replace('\\', '/')
		img = self._load_input_image(fullPathName)
		img = self.resize(img)

		if self.input_preprocess is not None:
			img = self.input_preprocess(img)

		# img = augmentation(img)
		img = self.to_tensor(img)
		img = self.normalize(img)

		# get mask
		fullPathName = os.path.join(self.mask_dir, idx)
		fullPathName = fullPathName.replace('\\', '/').split('.')[:-1]
		############################我这里改了 为了适应新的val###########
		# fullPathName = '.'.join(fullPathName)+'_Segmentation.png'
		fullPathName = '.'.join(fullPathName)+'_lesion.bmp'

		try:
			final_path = glob.glob(fullPathName)
			Mask = self._load_target_image(final_path[0])# Lmode是单通道, 并不妨碍是0-255
			Mask = self.resize(Mask)
			Mask = np.array(Mask)
			Mask = np.where(Mask > 0.5, 1, 0)

		except:
			Mask = np.zeros((256,256))

		# if self.target_preprocess is not None:
		# 	Mask = self.target_preprocess(Mask)
		Mask = self.to_tensor(Mask)
		# print(Mask.size())

		return {'image': img, 'mask': Mask, 'name': idx}