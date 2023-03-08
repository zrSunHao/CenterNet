# 默认配置

class DefaultConfig(object):
    # 基本的配置
    data_root = 'D:/WorkSpace/Python/CenterNet/data/'                   # 数据集的存放路径
    models_root = './checkpoints'           # 模型存放路径
    train_dir= 'train2017/'                 # 训练集路径
    val_dir= 'val2017/'                     # 验证集路径
    anno_dir= 'annotations/'                # 标签路径
    test_img_dir = './test'                 # 测试图片存储目录
    test_out_dir = './outputs'              # 测试图片标记输出目录
    img_size = 512                          # 图像尺寸
    device = 'cuda'                         # 使用的设备 cuda/cpu

    # 工具配置信息
    vis_use = True                          # 是否使用 Visdom 可视化
    vis_env = 'CenterNet'                         # Visdom 的 env
    plot_every = 20                         # 每间隔 20 个batch，Visdom 画图一次

    # 训练相关的配置
    max_epoch = 100                         # 最大训练轮次
    cur_epoch = 1                           # 当前训练的轮次，用于中途停止后再次训练时使用
    save_every = 1                          # 每训练多少个 epoch，保存一次模型
    num_workers = 0                         # 多进程加载数据所用的进程数，默认为0，表示不使用多进程
    batch_size = 8                          # 每批次加载图像的数量
    lr = 1e-3                               # 学习率

    # 模型相关
    classes_num = 80                        # 图片类别数
    topk = 100                              # 
    net_path = None                         # 预训练的判别器模型路径

    coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    
    coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                        70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
