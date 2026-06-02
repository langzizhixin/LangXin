import cv2
from piq import psnr, ssim, FID
import face_alignment
from piq.feature_extractors import InceptionV3
from models import define_D
from loss import GANLoss
from models import Renderer  
import argparse

# 修改图像尺寸为 512
img_size = 512 

parser = argparse.ArgumentParser()
parser.add_argument('--sketch_root', required=True, help='root path for sketches')
parser.add_argument('--face_img_root', required=True, help='root path for face frame images')
parser.add_argument('--audio_root', required=True, help='root path for audio mel')
args = parser.parse_args()

# 其他参数
num_workers = 4
Project_name = 'renderer_T1_ref_N3'  # Project_name
finetune_path = None
ref_N = 3
T = 1
print('Project_name:', Project_name)
batch_size = 2  # batch_size
batch_size_val = 2  # batch_size
