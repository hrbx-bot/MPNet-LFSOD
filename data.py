import os
import random

import numpy
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import cv2
from PIL import ImageEnhance


# rgbdirpath = r"D:\heqian\dataset\focal_stack\test_in_train\0000.jpg"
# focaldirpath = "D:\heqian\dataset\\focal_stack\\test_in_train\\0003.jpg"
# gtdirpath = r"D:\heqian\dataset\focal_stack\test_in_train\0002.jpg"


def cv_random_flip(img, label, mv):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(mv.shape[2]):
            mv[:,:,i] = numpy.flip(mv[:,:,i],1)
        # img.save(rgbdirpath)
        # label.save(gtdirpath)
        # focal_focal = focal[:, :, 0:3]
        # # print(focal_focal.shape)
        # cv2.imwrite(focaldirpath, focal_focal)
    return img, label, mv


def randomCrop(image, label, mv):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    W1 = (image_width - crop_win_width) >> 1
    H1 = (image_height - crop_win_height) >> 1
    W2 = (image_width + crop_win_width) >> 1
    H2 = (image_height + crop_win_height) >> 1
    random_region = (W1, H1, W2, H2)   #(2, 11, 253, 244) 与左边界的距离，与上边界的距离，与左边界的距离，与上边界的距离
    # print(random_region)
    # print(focal.shape)
    mv_crop = mv[H1:H2, W1:W2, :]
    image = image.crop(random_region)
    label = label.crop(random_region)
    # print(focal_crop.shape)
    # print(image.size)
    # image.save(rgbdirpath)
    # # label.save(gtdirpath)
    # focal_focal = focal_crop[:, :, 0:3]
    # # print(focal_focal.shape)
    # cv2.imwrite(focaldirpath, focal_focal)
    return image, label, mv_crop


def randomRotation(image, label, mv):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        # random_angle = np.random.randint(-15, 15)
        angle = [90,180,270]
        random_angle = random.choice(angle)
        if random_angle == 90:
            m = 1
        elif random_angle == 180:
            m = 2
        else:
            m = 3
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        for i in range(mv.shape[2]):
            mv[:, :, i] = np.rot90(mv[:, :, i],m)
        # image.save(rgbdirpath)
        # label.save(gtdirpath)
        # focal_focal = focal[:, :, 0:3]
        # cv2.imwrite(focaldirpath, focal_focal)
    return image, label, mv


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image



def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)






class SalObjDataset(data.Dataset):
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])

    # mean_focal = mean_rgb  # 现在直接使用 mean_rgb
    # std_focal = std_rgb  # 现在直接使用 std_rgb
    num_mv = 4  # optical flow 图像的数量 commented by lzg on 20250520
    mean_mv = np.tile(mean_rgb, num_mv)  # 将 mean_rgb 复制 4 倍
    std_mv = np.tile(std_rgb, num_mv)  # 将 std_rgb 复制 4 倍
    def __init__(self, image_root, gt_root, mv_root, trainsize):
        print(f"image_root: {image_root}")  # 调试信息
        print(f"gt_root: {gt_root}")  # 调试信息
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        # # 针对focal stack
        # self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]

        # # 针对单张optical flow
        # self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.png')]  # 更改为.png

        print(f"len(self.images): {len(self.images)}")  # 调试信息
        print(f"len(self.gts): {len(self.gts)}")  # 调试信息
        # print(f"len(self.focals): {len(self.focals)}")  # 调试信息
        # 针对多张optical flow
        self.mv_1_root = mv_root + 'of_1_2/'
        self.mv_2_root = mv_root + 'of_1_3/'
        self.mv_3_root = mv_root + 'of_1_4/'
        self.mv_4_root = mv_root + 'of_1_5/'

        # self.focal_1_root = focal_root + 'test_images/'
        # self.focal_2_root = focal_root + 'test_images/'
        # self.focal_3_root = focal_root + 'test_images/'
        # self.focal_4_root = focal_root + 'test_images/'

        self.mvs_1 = sorted([f for f in os.listdir(self.mv_1_root) if f.endswith('.png')])
        self.mvs_2 = sorted([f for f in os.listdir(self.mv_2_root) if f.endswith('.png')])
        self.mvs_3 = sorted([f for f in os.listdir(self.mv_3_root) if f.endswith('.png')])
        self.mvs_4 = sorted([f for f in os.listdir(self.mv_4_root) if f.endswith('.png')])

        #self.focals = [self.focal_1_root + f for f in os.listdir(self.focal_1_root) if f.endswith('.png')] #删除
        # self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.focals = sorted(self.focals) # 排序
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        # # 针对单张optical flow
        # # 定义 focal 图像的预处理流程 (focal_transform)，针对单张optical flow
        # self.focal_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor(),  # 转换为 Tensor 并归一化到 [0, 1]
        #     transforms.Normalize(self.mean_focal, self.std_focal),  # 标准化
        #     # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x) #如果需要1通道变3通道
        # ])


    #   return image, mask
    def __getitem__(self, index):
        # 1、加载原始图像
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # 针对多张optical flow
        mv = self.mv_loader(index)
        # # 针对mat和单张optical flow
        # focal = self.focal_loader(self.focals[index])

        # 2、统一尺寸
        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
        mv = cv2.resize(mv, (self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        # 如果 focal resize 后维度错了，比如变成了 (H, W)，需要reshape
        if mv.ndim == 2:
            # OpenCV 对于单通道输入可能会降维，需要恢复
            mv = mv.reshape(self.trainsize, self.trainsize, -1)

        # 3、数据增强
        image, gt, mv = cv_random_flip(image, gt, mv)
        image, gt, mv = randomRotation(image, gt, mv)
        image, gt, mv = randomCrop(image, gt, mv)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        # 4、数据增强完再次统一尺寸
        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
        mv = cv2.resize(mv, (self.trainsize, self.trainsize), interpolation=cv2.INTER_LINEAR)
        # 如果 focal resize 后维度错了，比如变成了 (H, W)，需要reshape
        if mv.ndim == 2:
            # OpenCV 对于单通道输入可能会降维，需要恢复
            mv = mv.reshape(self.trainsize, self.trainsize, -1)

        

        # # 针对单张optical flow
        # focal = self.focal_transform(focal)  # 使用focal_transform
        # # 针对mat和多张optical flow
        # focal = np.array(focal, dtype=np.int32)
        # # 检查FocalStack数据的高度是否为256。
        # # 如果高度不为256，则将每个Focal Slice的图像大小调整为256x256，并将所有Focal Slice图像沿着通道维度拼接在一起。
        # if focal.shape[0] != 256:
        #     new_focal = []
        #     focal_num = focal.shape[2] // 3
        #     for i in range(focal_num):
        #         a = focal[:, :, i * 3:i * 3 + 3].astype(np.uint8)
        #         a = cv2.resize(a, (256, 256))
        #         new_focal.append(a)
        #     focal = np.concatenate(new_focal, axis=2)  # (256, 256, 36)

        # 5、转换为tensor
        image = self.img_transform(image) #torch.Size([3, 256, 256])
        gt = self.gt_transform(gt) #torch.Size([1, 256, 256])
        # 将图片转化为numpy数组
        mv = mv.astype(np.float64)/255.0
        mv -= self.mean_mv
        mv /= self.std_mv
        mv = mv.transpose(2, 0, 1)
        # 把数组转换成张量，且二者共享内存
        mv = torch.from_numpy(mv).float() #torch.Size([36, 256, 256])

        return image, gt, mv
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    # 针对单张optical flow
    # def focal_loader(self, path):
    #     with open(path, 'rb') as f:
    #         img = Image.open(f)
    #         img = img.convert('RGB')  # 确保转换为 RGB
    #         assert len(img.getbands()) == 3, f"Focal image should have 3 channels, but got {len(img.getbands())}"
    #         return img


    # 针对多张optical flow
    def mv_loader(self, index):
        mv_paths = [
            os.path.join(self.mv_1_root, self.mvs_1[index]),
            os.path.join(self.mv_2_root, self.mvs_2[index]),
            os.path.join(self.mv_3_root, self.mvs_3[index]),
            os.path.join(self.mv_4_root, self.mvs_4[index]),
        ]
        mvs = []
        for path in mv_paths:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = np.array(img)
                mvs.append(img)
        mv = np.concatenate(mvs, axis=2)

        return mv

    # 针对mat文件
    # def focal_loader(self, path):
    #     with open(path, 'rb') as f:
    #         focal = sio.loadmat(f)
    #         focal get_loader= focal['img'] #上面函数返回的一键值对，选取键为’img‘的值给focal
    #         return focal

    # 针对mat和多张optical flow
    def resize(self, img, gt, mv):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), mv
        else:
            return img, gt, mv

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, mv_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True): #pin_memory设置为True可以将Tensor放入到内存的锁页区，加快训练速度

    dataset = SalObjDataset(image_root, gt_root, mv_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    num_mv = 4  # Multi-focal 图像的数量
    # mean_focal = mean_rgb  # 现在直接使用 mean_rgb
    # std_focal = std_rgb  # 现在直接使用 std_rgb
    mean_mv = np.tile(mean_rgb, num_mv)  # 将 mean_rgb 复制 4 倍
    std_mv = np.tile(std_rgb, num_mv)  # 将 std_rgb 复制 4 倍
    def __init__(self, image_root, gt_root, mv_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]

        # # 针对单张optical flow
        # self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.png')]  # 更改为.png

        # 针对多张optical flow
        self.mv_1_root = mv_root + 'of_1_2/'
        self.mv_2_root = mv_root + 'of_1_3/'
        self.mv_3_root = mv_root + 'of_1_4/'
        self.mv_4_root = mv_root + 'of_1_5/'


        

        self.mvs_1 = sorted([f for f in os.listdir(self.mv_1_root) if f.endswith('.png')
                                or f.endswith('.jpg')])
        self.mvs_2 = sorted([f for f in os.listdir(self.mv_2_root) if f.endswith('.png')
                                or f.endswith('.jpg')])
        self.mvs_3 = sorted([f for f in os.listdir(self.mv_3_root) if f.endswith('.png')
                                or f.endswith('.jpg')])
        self.mvs_4 = sorted([f for f in os.listdir(self.mv_4_root) if f.endswith('.png')
                                or f.endswith('.jpg')])


        # self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        print(f"len(self.images): {len(self.images)}")  # 调试信息
        print(f"len(self.gts): {len(self.gts)}")  # 调试信息
        # print(f"len(self.focals): {len(self.focals)}")  # 调试信息
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.focals = sorted(self.focals)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        # # 针对单张optical flow
        # self.focal_transform = transforms.Compose([  # 可选：为focal图像添加transform
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor(),  # 这一步会将focal图归一化到[0, 1]
        #     transforms.Normalize(self.mean_focal, self.std_focal),
        #     # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x) #如果需要1通道变3通道
        # ])
        self.size = len(self.images)
        self.index = 0
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])

        # # 处理mat和单张optical flow
        # focal = self.focal_loader(self.focals[self.index])  # focal现在是PIL Image
        # focal = self.focal_transform(focal)  # 应用focal_transform

        # 处理多张optical flow
        mv = self.mv_loader(self.index) #focal.shape = (256, 256, 256)

        # 针对mat和多张optical flow
        mv = np.array(mv, dtype=np.int32)
        if mv.shape[0] != 256:
            new_mv = []
            mv_num = mv.shape[2] // 3
            for i in range(mv_num):
                a = mv[:, :, i * 3:i * 3 + 3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_mv.append(a)
            mv = np.concatenate(new_mv, axis=2)

        mv = mv.astype(np.float64)/255.0
        mv -= self.mean_mv
        mv /= self.std_mv
        mv = mv.transpose(2, 0, 1)
        mv = torch.from_numpy(mv).float()


        name = self.images[self.index].split('/')[-1]
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, mv, gt, name


    # single optical flow
    # def focal_loader(self, path):
    #     with open(path, 'rb') as f:
    #         img = Image.open(f)
    #         img = img.convert('RGB')  # 确保转换为 RGB
    #         assert len(img.getbands()) == 3, f"Focal image should have 3 channels, but got {len(img.getbands())}"
    #         return img

    # focal stack mat
    # def focal_loader(self, path):
    #     with open(path, 'rb') as f:
    #         focal = sio.loadmat(f)
    #         focal = focal['img']
    #         return focal

    # 多张 optical flow
    def mv_loader(self, index):
        mv_paths = [
            os.path.join(self.mv_1_root, self.mvs_1[index]),
            os.path.join(self.mv_2_root, self.mvs_2[index]),
            os.path.join(self.mv_3_root, self.mvs_3[index]),
            os.path.join(self.mv_4_root, self.mvs_4[index]),
        ]
        mvs = []
        for path in mv_paths:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = np.array(img)
                mvs.append(img)
        mv = np.concatenate(mvs, axis=2)

        return mv

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


