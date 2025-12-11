import logging
import os
from datetime import datetime
import numpy as np
from options import opt
from torch.optim.lr_scheduler import CosineAnnealingLR

if opt.gpu_id == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('USE GPU 1')

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
# from torch.utils.tensorboard import SummaryWriter
from data import get_loader, test_dataset
from model.MPNet import model

from utils import clip_gradient, adjust_lr
from tools.pytorch_utils import Save_Handle
from torch.autograd import Variable


# # ---------------------------------------------------------tensorboard性能分析-----------------------------------------------------------------
# import torch.profiler # <--- 确保导入
# # --- 在 train 函数外部创建 profiler 日志目录 ---
# profiler_log_dir = "./log/profiler"
# if not os.path.exists(profiler_log_dir):
#     os.makedirs(profiler_log_dir)
# ---------------------------------------------------------tensorboard性能分析-----------------------------------------------------------------



torch.cuda.current_device()

print("GPU available:", torch.cuda.is_available())

cudnn.benchmark = True
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_list = Save_Handle(max_num=1)


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()#numel用于返回数组中的元素个数
    print(name)
    print('The number of parameters:{}'.format(num_params))
    return num_params
start_epoch = 0

model = model()
if (opt.load_mit is not None):
    model.of_encoder_shared.init_weights(opt.load_mit)
    model.rgb_encoder.init_weights(opt.load_mit)
else:
    print("No pre-trian!")

model.cuda()


# # commented by lzg on 20250522
# # --- 在这里添加 torch.compile ---
# if hasattr(torch, 'compile'): # 检查PyTorch版本是否支持
#     print("Compiling model with torch.compile()...")
#     try:
#         model = torch.compile(model) # 默认模式
#         print("Model compiled successfully.")
#     except Exception as e:
#         print(f"Failed to compile model: {e}")
#         print("Continuing without compiling.")
# # --- 添加结束 ---

# model.to(device)
params = model.parameters()
optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  #weight_decay正则化系数
model_params = print_network(model, 'lf_pvt')
rgb_root = opt.rgb_root
gt_root = opt.gt_root
mv_root = opt.mv_root
test_rgb_root = opt.test_rgb_root
test_mv_root = opt.test_mv_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path



if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(rgb_root, gt_root, mv_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_rgb_root, test_gt_root, test_mv_root,testsize=opt.trainsize)
total_step = len(train_loader)
logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("Net-Train")
logging.info("Config")
logging.info('params:{}'.format(model_params))
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,  save_path,
        opt.decay_epoch))


#set loss function
# CE = torch.nn.BCEWithLogitsLoss()


def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


step = 0
# writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    # --- 1. 定义 Profiler 上下文管理器 tensorboard---
    # schedule: wait=1, warmup=1, active=3, repeat=1 -> 每个epoch分析5个step
    #           (1个等待, 1个预热, 3个记录)，只重复一次
    # on_trace_ready: 每次记录周期结束后，都会调用这个handler保存追踪文件
    #               文件名会包含 step number，所以不会相互覆盖
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
    #     record_shapes=True,
    #     with_stack=True
    # )

    # --- 2. 使用 `with prof:` 包裹整个try和循环，实现tensorboard性能分析 ---
    # with prof:
    try:
        for i, (images, gts, mv) in enumerate(train_loader, start=1):
            basize, dim, height, width = mv.size()
            gts = gts.cuda()
            images, gts, mv = Variable(images), Variable(gts), Variable(mv)
            gts1 = F.interpolate(gts, size=(64, 64), mode='bilinear', align_corners=False)
            gts2 = F.interpolate(gts, size=(32, 32), mode='bilinear', align_corners=False)
            gts3 = F.interpolate(gts, size=(16, 16), mode='bilinear', align_corners=False)
            gts4 = F.interpolate(gts, size=(8, 8), mode='bilinear', align_corners=False)
            mv = mv.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            # commented by lzg on 2025.6.5
            mv = torch.cat(torch.chunk(mv, chunks=4, dim=2), dim=1)  # (basize, 12, 3, 256, 256)

            mv = torch.cat(torch.chunk(mv, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            mv = mv.view(-1, *mv.shape[2:])  # [basize*12, 6, 256, 256)
            mv = mv.cuda()
            images = images.cuda()
            optimizer.zero_grad()
            # outputs = model(focal, images)  #commented by lzg on 2024.12.17
            # print("Model outputs:", outputs)    #commented by lzg on 2024.12.17
            # print("focal.shape:",focal.shape)   #commented by lzg on 2025.2.22


            x1, x2, x3, x4, mv_sal, _, _ , _, _ = model(mv, images)   #commented by lzg on 2024.12.17
            loss = structure_loss(mv_sal, gts) + structure_loss(x1, gts1) + structure_loss(x2, gts2) + structure_loss(x3, gts3) + structure_loss(x4, gts4)
            loss.backward()
            # 梯度裁剪
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||Mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], loss.data, memory_used))
        
        # --- 3. 在循环末尾调用 prof.step()   tensorboard性能分析---
            # Profiler 会根据 schedule 自动决定当前 step 是 wait, warmup, 还是 active
            # prof.step()

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))

        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'lfsod_epoch_{}.pth'.format(epoch))  #每5轮保存一次
        # torch.save(model.state_dict(), save_path + 'lfsod_epoch_{}.pth'.format(epoch))  #每一轮都保存

        # 训练中断保留参数
        temp_save_path = save_path + "{}_ckpt.tar".format(epoch)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, temp_save_path)
        save_list.append(temp_save_path)

    except KeyboardInterrupt:
        logging.info('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'lfnet_epoch_{}.pth'.format(epoch + 1))
        logging.info('save checkpoints successfully!')
        raise





def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, mv, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            dim, height, width = mv.size()
            basize = 1


            mv = mv.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            mv = torch.cat(torch.chunk(mv, chunks=4, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
            mv = torch.cat(torch.chunk(mv, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            mv = mv.view(-1, *mv.shape[2:])
            mv = mv.cuda()
            image = image.cuda()

            # print("focal.shape:",focal.shape)   #commented by lzg on 2025.2.22

            _, _, _, _, res, _, _, _, _ = model(mv, image)


            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        logging.info('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch

                # 应对torch.compile过的model与test.py中创建的model不匹配而添加的保存原始model的代码 2025.6.8
                # 获取原始模型 (如果 model 是被 torch.compile 包装的)
                model_to_save = model
                # if hasattr(model, '_orig_mod'):  # 检查是否有 _orig_mod 属性
                #     model_to_save = model._orig_mod

                torch.save(model_to_save.state_dict(), save_path + 'lfsod_epoch_best.pth')

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    logging.info("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    # scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr*0.1)
    for epoch in range(start_epoch, opt.epoch+1):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)

        # --- NEW: 在 epoch 的末尾，更新学习率 ---
        # scheduler.step() 会根据当前是第几个 epoch，
        # 按照余弦退火的公式，为 optimizer 计算并设置下一个 epoch 的新学习率。
        # scheduler.step()





































