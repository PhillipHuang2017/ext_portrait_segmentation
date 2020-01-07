import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import glob

import json

from PIL import Image as PILImage
import importlib
from torchvision.transforms import functional as F

import pickle

from argparse import ArgumentParser
from etc.utils import *
# torch.backends.cudnn.benchmark=True

pallete = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


# 参数imgW, imgH为网络的输入尺寸
def evaluateModelCV(model, inputDir, outputDir, mean, std, imgW, imgH, videoName, Lovasz):
    # gloabl mean and std values

    videoOut = os.path.join(outputDir,videoName)
    videoIn = os.path.join(inputDir,videoName)
    
    print("video is saved in " + videoOut)

    # 输出视频尺寸为输入视频尺寸
    vidcap = cv2.VideoCapture(videoIn)
    videoW = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoH = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(videoOut, cv2.VideoWriter_fourcc(*'mp4v'), 30, (videoW, videoH))
    # syn_bg = np.zeros((videoH, videoW, 3), dtype=np.uint8)   # opecv中3通道图像维度是(h, w, 3)，通过shape属性可以看到
    all_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame = 0
    success= True
    init_time = time.clock()
    while success:
        success, img = vidcap.read()
        if not success or img is None:
            vidcap.release()
            break

        img_orig = np.copy(img)
        # PILImage.fromarray(img_orig).show()

        img = cv2.resize(img, (imgW, imgH))
        # PILImage.fromarray(img).show()

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        img /= 255
        img = img.transpose((2, 0, 1))   # 维度交换，使其变成通道维度在前
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension

        with torch.no_grad():
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()

            img_out = model(img_tensor)
        # img_orig = cv2.resize(img_orig, (imgW, imgH))
        if Lovasz:
            classMap_numpy = (img_out[0].cpu() > 0).numpy()[0]
        else:
            # 取概率大的那个类作为判别结果，下标0或者1就可以表示人或者背景类
            classMap_numpy = img_out[0].max(0)[1].byte().cpu().numpy()

        # 先升维成(w, h, 1)，然后把最后一维重复3次，复制成3通道掩码，
        # 因为如果没有第3维的话和图像相乘的时候不能自动广播
        # 注意一定要变成uint8类型，不然无法用cv2.resize
        idx_fg = classMap_numpy[:, :, np.newaxis].astype(np.uint8).repeat(3, axis=2)
        # 把掩码缩放至原图尺寸
        idx_fg = cv2.resize(idx_fg, (videoW, videoH))

        seg_img = img_orig*idx_fg

        frame += 1
        fps = frame/(time.clock() - init_time)

        # 显示fps
        fps_text = ' %5.2f fps '%(fps)
        font_face = cv2.FONT_HERSHEY_DUPLEX  # 字体
        font_scale = 0.6  # 字体大小
        font_thickness = 1 # 字体粗细
        textw, texth = cv2.getTextSize(fps_text, font_face, font_scale, font_thickness)[0]
        seg_img = seg_img.astype(np.float)  # 先转成float，不然无法和0.6相乘
        seg_img[0:texth+8, videoW-textw-9:] *= 0.6  # 准备把fps信息放在右上角
        seg_img = seg_img.astype(np.uint8)

        text_point = (videoW-textw-5, texth+4)  # 文字左下角坐标
        text_color = [255, 255, 255]
        cv2.putText(seg_img, fps_text, text_point, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        video.write(seg_img)
        print('\rProcessing frames %6d / %6d (%5.2f%%)  fps: %5.2f'%(frame, all_frames, 100*(frame/all_frames), fps), end='')

    video.release()
    # cv2.destroyAllWindows()


# 这里我把参数的顺序调整了，可能导致其他地方调用的时候出问题
# 这里的h, w参数表示送入网络时的数据尺寸
def ExportVideo(model, model_name, weightPath, inputDir, videoName, outputDir, w, h, mean, std, Lovasz):
    # read all the images in the folder

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weightPath))
    else:
        model.load_state_dict(torch.load(weightPath,"cpu"))

    model.eval()


    if not os.path.isdir(inputDir):
        os.makedirs(inputDir)
    
    outputDir = os.path.join(outputDir,model_name)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    
    evaluateModelCV(model, inputDir, outputDir, mean, std, w, h, videoName, Lovasz)

if __name__ == '__main__':
    import models


    parser = ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default='./setting/Test_SINet.json',
                        help='JSON file for configuration')
    parser.add_argument('--outputdir', type=str, default="./segVideos/")
    parser.add_argument('--weight', type=str, default="./result/SINet/SINet.pth")
    parser.add_argument('--video', type=str, default="./videos/cy_car_0523.mp4")
    parser.add_argument('--inputW', type=int, default=224)
    parser.add_argument('--inputH', type=int, default=224)
    parser.add_argument('--input_size', type=int, default=0)

    args = parser.parse_args()
    if args.input_size > 0:
        inputW = args.input_size
        inputH = args.input_size
    else:
        inputW = args.inputW
        inputH = args.inputH

    #weightPath = "../result/Dnc_SINet11-24_2218/model_3.pth"
    weightPath = args.weight
    videoDir, videoName = os.path.split(args.video)
    outputDir = args.outputdir

    mean = [107.304565, 115.69884, 132.35703 ]
    std = [63.97182, 65.1337, 68.29726]
    
    with open(args.config) as fin:
        config = json.load(fin)

    train_config = config['test_config']
    data_config = config['data_config']

    model_name = train_config['Model']

    Lovasz = train_config["loss"] == "Lovasz"
    # 训练Encoder的时候用的是"Lovasz"损失，测试的时候用Decoder测试，用的CE损失
    if Lovasz:
        train_config["num_classes"] = train_config["num_classes"] -1
    if 'SINet' in model_name:
        model = models.__dict__[model_name](classes=train_config["num_classes"],
                                            p=train_config["p"], q=train_config["q"], chnn=train_config["chnn"])
    elif 'ExtremeC3Net' in model_name:
        model = models.__dict__[model_name](classes=train_config["num_classes"],
                                            p=train_config["p"], q=train_config["q"], stage1_W=train_config['stage1_W'])

    if torch.cuda.device_count() > 0:
        model=model.cuda()

    ExportVideo(model, model_name, weightPath, videoDir, videoName, outputDir, inputW, inputH, mean, std, Lovasz)
