import os
import torch as t
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter

from models import CenterNet
from dataproviders import COCODataset,Augmentation
from configs import DefaultCfg 
from tools import Visualizer, detection_collate,gt_creator

# 配置
cfg = DefaultCfg()
device = t.device(cfg.device)
vis = Visualizer(cfg.vis_env)

# 数据
dataset = COCODataset(data_dir = cfg.data_root,
                      anno_dir = cfg.anno_dir,
                      name = cfg.train_dir,
                      transform=Augmentation(img_size = cfg.img_size)
                      )
dataloader = DataLoader(dataset = dataset,
                        batch_size = cfg.batch_size,
                        shuffle = True,
                        collate_fn = detection_collate,
                        num_workers = cfg.num_workers
                        )

# 模型
model_ft = CenterNet(classes_num = cfg.classes_num,
                     topk = cfg.topk)
if not cfg.net_path is None:
    model_path = os.path.join(cfg.models_root, cfg.net_path)
    if os.path.exists(model_path):
        state_dict = t.load(model_path)
        model_ft.load_state_dict(state_dict)
model_ft.to(device)

# 优化器
optimizer = t.optim.Adam(model_ft.parameters(), cfg.lr)

loss_meter = AverageValueMeter()
epochs = range(cfg.max_epoch)
for epoch in iter(epochs):

    if epoch+1 < cfg.cur_epoch:
        continue
    loss_meter.reset()

    for ii, (imgs, target) in enumerate(dataloader):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        target = [label.tolist() for label in target]
        target = gt_creator(input_size = cfg.img_size,
                            stride = 4,
                            classes_num = cfg.classes_num,
                            label_list = target)
        target = t.tensor(target).float().to(device=device)

        total_loss = model_ft(imgs, target)
        total_loss.backward()
        optimizer.step()
        loss_meter.add(total_loss.item())

        if (ii+1) % cfg.plot_every == 0:
            vis.plot('total_loss', loss_meter.value()[0])
        print('%s / %s'%(ii, len(dataset)/cfg.batch_size))

    vis.save([cfg.vis_env])
    if (epoch+1) % cfg.save_every == 0:
        model_path = '%s/centernet_%s.pth'% (cfg.models_root, str(epoch+1))
        t.save(model_ft.state_dict(), model_path)