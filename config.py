import math


class DefaultConfigs(object):
    seed = 666
    visualization_path = './visualizations'  # 可视化结果保存路径
    vis_red_band = 29  # 伪彩图红色通道波段
    vis_green_band = 19  # 伪彩图绿色通道波段
    vis_blue_band = 9  # 伪彩图蓝色通道波段
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    # training parameters
    train_epoch = 100
    test_epoch = 5
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    # source data information
    data = 'PaviaU'  # PaviaU-9 /  Houston2018-20
    num_classes = 9
    patch_size = 13
    pca_components = 30
    test_ratio = 0.95
    # model

    model_type = 'DEM'

    depth = 1
    embed_dim = 32
    dim_inner = 2*embed_dim
    # dim_inner = embed_dim
    dt_rank = math.ceil(embed_dim/16)
    d_state = 16
    group_type = 'Cube'  # Linear  Patch  Cube
    scan_type = 'Parallel spectral-spatial'  #Spectral-priority  #Spatial-priority  #Cross spectral-spatial  Parallel spectral-spatial #spatial-spectral   spectral-spatial   spectral  spatial
    k_group = 4
    pos = False
    cls = False
    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel = (3, 5, 5)
    dim_patch = patch_size - conv3D_kernel[1] + 1  # 8
    dim_linear = pca_components - conv3D_kernel[0] + 1  # 28

    # dendritic network parameter
    dendritic_in_channel = 32
    dendritic_out_channel = 9  # 类别数
    num_branch = 8
    # synapse_activation = "Softmax"
    # dendritic_activation = "None"
    # soma = "None"

    # paths information
    checkpoint_path = './' + "checkpoint/" + data + '/' + model_type + '/' + 'TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN) \
                      + '/Pos(' + str(pos) + ')' + '_Cls(' + str(cls)+ ')_' + scan_type + '/PatchSize' + str(patch_size) + '_numbranch' + str(num_branch) + '_TestRatio' + str(test_ratio) \
                      + '/' + group_type + str(k_group) + '_depth' + str(depth) + '_embed' + str(embed_dim) + '_dtrank' + str(dt_rank) + '_dstate' + str(d_state) + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel) + '/'
    logs = checkpoint_path


config = DefaultConfigs()
