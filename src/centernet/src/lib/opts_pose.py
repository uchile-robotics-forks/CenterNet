from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opt_struct:
    def __init__(self):
        # basic experiment setting
        self.task = 'multi_pose'
        self.dataset = 'coco'
        self.load_model = '../models/multi_pose_dla_3x.pth'
        self.exp_id = 'default'
        self.test = False
        self.debug = 0
        self.demo = 'webcam'
        self.resume = False
    

        # system
        self.gpus = '0'

        self.num_workers = 4
    # self.parser.add_argument('--not_cuda_benchmark', action='store_true',
    #                          help='disable when the input size is not fixed.')
    # self.parser.add_argument('--seed', type=int, default=317, 
    #                          help='random seed') # from CornerNet

        # log
    # self.parser.add_argument('--print_iter', type=int, default=0, 
    #                          help='disable progress bar and print to screen.')
    # self.parser.add_argument('--hide_data_time', action='store_true',
    #                          help='not display time during training.')
    # self.parser.add_argument('--save_all', action='store_true',
    #                          help='save model to disk every 5 epochs.')
        self.metric = 'loss'
        self.vis_thresh = 0.3
        self.debugger_theme ='white' 
    
        # input
        self.input_res = -1
        self.input_h = -1 
        self.input_w = -1 
    
        
        # model
        self.arch = 'dla_34'
        self.head_conv = -1
        self.down_ratio = 4

        #train
        self.lr = '1.25e-4'
        self.lr_step = '90,120'
        self.num_epochs = 140
        self.batch_size = 32
        self.master_batch_size = -1
        self.num_iters = -1
        self.val_intervals = 5
        self.trainval = False
        
        #test
        self.flip_test = 'store_true'
        self.test_scales = '1'
        self.nms = False
        self.K = 100
        self.not_prefetch_test = False
        self.fix_res =False
        self.keep_res = False

    # # dataset
    # self.parser.add_argument('--not_rand_crop', action='store_true',
    #                          help='not use the random crop data augmentation'
    #                               'from CornerNet.')
    # self.parser.add_argument('--shift', type=float, default=0.1,
    #                          help='when not using random crop'
    #                               'apply shift augmentation.')
    # self.parser.add_argument('--scale', type=float, default=0.4,
    #                          help='when not using random crop'
    #                               'apply scale augmentation.')
    # self.parser.add_argument('--rotate', type=float, default=0,
    #                          help='when not using random crop'
    #                               'apply rotation augmentation.')
    # self.parser.add_argument('--flip', type = float, default=0.5,
    #                          help='probability of applying flip augmentation.')
    # self.parser.add_argument('--no_color_aug', action='store_true',
    #                          help='not use the color augmenation '
    #                               'from CornerNet')
    
        # multi_pose
        self.aug_rot = 0

    # # ddd
    # self.parser.add_argument('--aug_ddd', type=float, default=0.5,
    #                          help='probability of applying crop augmentation.')
    # self.parser.add_argument('--rect_mask', action='store_true',
    #                          help='for ignored object, apply mask on the '
    #                               'rectangular region or just center point.')
    # self.parser.add_argument('--kitti_split', default='3dop',
    #                          help='different validation split for kitti: '
    #                               '3dop | subcnn')

        # loss
        self.mse_loss = False

    # # ctdet
    # self.parser.add_argument('--reg_loss', default='l1',
    #                          help='regression loss: sl1 | l1 | l2')
    # self.parser.add_argument('--hm_weight', type=float, default=1,
    #                          help='loss weight for keypoint heatmaps.')
    # self.parser.add_argument('--off_weight', type=float, default=1,
    #                          help='loss weight for keypoint local offsets.')
    # self.parser.add_argument('--wh_weight', type=float, default=0.1,
    #                          help='loss weight for bounding box size.')
    # # multi_pose
    # self.parser.add_argument('--hp_weight', type=float, default=1,
    #                          help='loss weight for human pose offset.')
    # self.parser.add_argument('--hm_hp_weight', type=float, default=1,
    #                          help='loss weight for human keypoint heatmap.')
    # # ddd
    # self.parser.add_argument('--dep_weight', type=float, default=1,
    #                          help='loss weight for depth.')
    # self.parser.add_argument('--dim_weight', type=float, default=1,
    #                          help='loss weight for 3d bounding box size.')
    # self.parser.add_argument('--rot_weight', type=float, default=1,
    #                          help='loss weight for orientation.')
    # self.parser.add_argument('--peak_thresh', type=float, default=0.2)
    
    # # task
    
    # # exdet
    # self.parser.add_argument('--agnostic_ex', action='store_true',
    #                          help='use category agnostic extreme points.')
    # self.parser.add_argument('--scores_thresh', type=float, default=0.1,
    #                          help='threshold for extreme point heatmap.')
    # self.parser.add_argument('--center_thresh', type=float, default=0.1,
    #                          help='threshold for centermap.')
    # self.parser.add_argument('--aggr_weight', type=float, default=0.0,
    #                          help='edge aggregation weight.')
    

    # # ground truth validation
    # self.parser.add_argument('--eval_oracle_hm', action='store_true', 
    #                          help='use ground center heatmap.')
    # self.parser.add_argument('--eval_oracle_wh', action='store_true', 
    #                          help='use ground truth bounding box size.')
    # self.parser.add_argument('--eval_oracle_offset', action='store_true', 
    #                          help='use ground truth local heatmap offset.')
    # self.parser.add_argument('--eval_oracle_kps', action='store_true', 
    #                          help='use ground truth human pose offset.')
    # self.parser.add_argument('--eval_oracle_hmhp', action='store_true', 
    #                          help='use ground truth human joint heatmaps.')
    # self.parser.add_argument('--eval_oracle_hp_offset', action='store_true', 
    #                          help='use ground truth human joint local offset.')
    # self.parser.add_argument('--eval_oracle_dep', action='store_true', 
    #                          help='use ground truth depth.')


        
        

        self.hp_weight = 1
        self.hm_hp_weight = 1


        # task
        # ctdet
        self.norm_wh = False
        self.dense_wh = False
        self.cat_spec_wh = False
        self.not_reg_offset =False

        #multi_pose        
        self.dense_hp = False
        self.not_hm_hp = False
        self.not_reg_hp_offset = False
        self.not_reg_bbox = False
        


class opts(object):
  def __init__(self):

    self.opt = opt_struct()
    
    
  def parse(self, args=None):
    
    opt = self.opt
      
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'exdet':
      # assert opt.dataset in ['coco']
      num_hm = 1 if opt.agnostic_ex else opt.num_classes
      opt.heads = {'hm_t': num_hm, 'hm_l': num_hm, 
                   'hm_b': num_hm, 'hm_r': num_hm,
                   'hm_c': opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg_t': 2, 'reg_l': 2, 'reg_b': 2, 'reg_r': 2})
    elif opt.task == 'ddd':
      # assert opt.dataset in ['gta', 'kitti', 'viper']
      opt.heads = {'hm': opt.num_classes, 'dep': 1, 'rot': 8, 'dim': 3}
      if opt.reg_bbox:
        opt.heads.update(
          {'wh': 2})
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'ctdet':
      # assert opt.dataset in ['pascal', 'coco']
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    elif opt.task == 'multi_pose':
      # assert opt.dataset in ['coco_hp']
      opt.flip_idx = dataset.flip_idx
      opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 34}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
      if opt.hm_hp:
        opt.heads.update({'hm_hp': 17})
      if opt.reg_hp_offset:
        opt.heads.update({'hp_offset': 2})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, args=None):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'exdet': {'default_resolution': [512, 512], 'num_classes': 80, 
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
      'multi_pose': {
        'default_resolution': [512, 512], 'num_classes': 1, 
        'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
        'dataset': 'coco_hp', 'num_joints': 17,
        'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                     [11, 12], [13, 14], [15, 16]]},
      'ddd': {'default_resolution': [384, 1280], 'num_classes': 3, 
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti'},
    }
    class Struct:
      def __init__(self, entries):
        self.default_resolution = [512, 512]
        self.num_classes = 1
        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]
        self.dataset = 'coco_hp'
        self.num_joints = 17
        self.flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
