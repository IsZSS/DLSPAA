task_name = 'ETrans'  # 给当前任务取一个自己能看懂的名字（只能用英文）
city = 'SIN'  # 切换城市 PHO, NYC, SIN
gpuId = "cuda:1"

LS_strategy = 'DoubleTrans'  # 长短期偏好学习策略 DoubleTrans

enable_fix_alpha = False  # 长短期偏好进行融合时，采用固定的alpha值
fix_alpha = 1.0  # 0.10.3 0.5 0.7 0.9

enable_alpha = True  # 是否自动学习长期偏好的权重

enable_user_embedding = True  # 是否采用自己编写得时空转移模式获取模式编码方法

enable_CL = True  # Wonder：contrastive learning

enable_filatt = False  # 以注意力的形式融合长短期偏好

if CL_strategy == 'BPR':
    CL_weight = 0.1
elif CL_strategy == 'Triplet':
    if city == 'PHO':
        CL_weight = 0.05
    elif city == 'NYC':
        CL_weight = 0.04
    elif city == 'SIN':
        CL_weight = 0.05
else:
    raise Exception('CL_strategy error')

enable_curriculum_learning = False  # Wonder: curriculum learning，模型训练过程中不断增大对比损失的权重，直到第 curriculum_num 个 epoch 时达到最大权重值 CL_max_weight
curriculum_num = 10
CL_max_weight = 0.2

tfp_layer_num = 5  # 控制transformer块

lr = 1e-5  # 1e-4 或 1e-5
if city == 'PHO':
    epoch = 40
    embed_size = 100  # 论文参数 PHO 100，NYC 40，SIN 60
    run_times = 5  # 整个模型训练的次数
    drop_steps = 1
elif city == 'NYC':
    epoch = 25
    embed_size = 40
    run_times = 5
    drop_steps = 2
elif city == 'SIN':  # 默认60
    epoch = 25
    embed_size = 60
    run_times = 5
    drop_steps = 2
else:
    raise Exception("City name error!")

output_file_name = f'{task_name}_{city}_{LS_strategy}_Epoch{epoch}_LR' + '{:.0e}'.format(lr)

if enable_drop:
    output_file_name = output_file_name + f'_Drop{drop_steps}'
else:
    output_file_name = output_file_name + "_NoDrop"

if enable_alpha:
    output_file_name = output_file_name + '_Alpha'

if enable_fix_alpha:
    output_file_name = output_file_name + f'_fix_{fix_alpha}'

if enable_filatt:
    output_file_name = output_file_name + f'_finall_att'  # 以注意力机制融合长短期偏好

if enable_user_embedding:
    output_file_name = output_file_name + f'_user_embed'

if enable_CL:
    if enable_curriculum_learning:
        output_file_name += f'_{CL_strategy}_{curriculum_num}_{CL_max_weight}'
    else:
        output_file_name += f'_{CL_strategy}{CL_weight}'
else:
    output_file_name = output_file_name + "_NoCL"

output_file_name = output_file_name + '_Embed' + str(embed_size) + f'_{tfp_layer_num}'
