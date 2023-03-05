# 默认配置

class DefaultConfig(object):
    # 基本的配置
    data_root = './data'                    # 数据集的存放路径
    models_root = './checkpoints'           # 模型存放路径
    train_dir= '/train2017'                 # 训练集路径
    val_dir= '/val2017'                     # 验证集路径
    label_dir= '/annotations_trainval2017'  # 标签路径
    test_out_dir = './outputs'              # 测试图片标记输出目录
    image_size = 96                         # 图像尺寸
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
