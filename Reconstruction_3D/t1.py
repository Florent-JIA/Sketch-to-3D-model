import torch
from collections import OrderedDict
import numpy as np
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import util.util_img

# net1_path = '/home/oem/Downloads/code/GenRe-ShapeHD/downloads/models/depth_pred_with_inpaint.pt'
# state_dicts = torch.load(net1_path)
# weights = state_dicts['nets'][0]
#
# new_weights = OrderedDict()
# for key, value in weights.items():
#     new_key = key.replace('net1.', '')  # 删除 "net1."
#     new_weights[new_key] = value
#
# print(type(state_dicts))
# print(state_dicts.keys())
# print(type(weights))
# print(len(weights))
# print(weights.keys())
# # print(weights)
# print()
#
# print(type(new_weights))
# print(len(new_weights))
# print(new_weights.keys())
# print(new_weights)

def mask(input_image, input_mask, bg=1.0):
    assert isinstance(bg, (int, float))
    assert (input_mask >= 0).all() and (input_mask <= 1).all()
    input_mask = input_mask.expand_as(input_image)
    bg = bg * input_image.new_ones(input_image.size())
    output = input_mask * input_image + (1 - input_mask) * bg
    return output

rgb_path = r'/home/oem/Downloads/code/GenRe-ShapeHD/output/batch0000/depth_image.png'
silhou_path = r'/home/oem/Downloads/code/GenRe-ShapeHD/downloads/data/test/genre/02958343_1a0bc9ab92c915167ae33d942430658c_view003_silhouette.png'

rgbImage = Image.open(rgb_path)
silhouImage = Image.open(silhou_path)

transform_to_tensor = transforms.Compose([transforms.ToTensor()])

rgbTensor = transform_to_tensor(rgbImage)
silhouTensor = transform_to_tensor(silhouImage)
# print(rgbTensor.shape)

# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(rgbTensor.permute(1, 2, 0))
# axs[0].axis('off')
# axs[1].imshow(silhouTensor.permute(1, 2, 0))
# axs[1].axis('off')
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()

print(rgbTensor.shape)
print(silhouTensor.shape)

rgb_silhou = mask(rgbTensor, silhouTensor)
# rgb_silhou = rgb_silhou.permute(1, 2, 0)
plt.imshow(rgb_silhou.permute(1, 2, 0))
plt.show()