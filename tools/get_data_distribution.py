import os, sys
import numpy as np
import matplotlib.pyplot as plt
from lib.datasets.kitti.kitti_utils import get_objects_from_label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)


def get_kitti_bev_distribution(root_dir='../../data',
                               split='trainval',
                               write_list=['Car']):

    assert split in ['train', 'val', 'trainval', 'test']
    split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
    idx_list = [x.strip() for x in open(split_dir).readlines()]
    data_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if split == 'test' else 'training')

    x_list = []
    z_list = []

    for idx in idx_list:
        label_dir = os.path.join(data_dir, 'label_2')
        label_file = os.path.join(label_dir, '%06d.txt' % int(idx))
        assert os.path.exists(label_file)
        objects = get_objects_from_label(label_file)
        for obj in objects:
            if obj.cls_type not in write_list: continue
            # if obj.get_obj_level() != 1: continue   # easy only
            # if obj.get_obj_level() != 2: continue   # moderate only
            # if obj.get_obj_level() != 3: continue   # hard only
            # if obj.get_obj_level() != 4: continue   # unknown only

            x_list.append(obj.pos[0])
            z_list.append(obj.pos[2])

    z = np.array(z_list)

    # print stats
    print ('all samples:', len(z_list))
    print ('samples > 60:', (z>60).sum())
    print ('samples > 65:', (z>65).sum())
    print ('samples < 5: ', (z<5).sum())
    print ('samples < 10:', (z<10).sum())
    print ('samples < 15:', (z<15).sum())
    print ('samples < 20:', (z<20).sum())
    print ('samples in [5,15]:', (z<15).sum() - (z<5).sum())
    print ('samples in [10,20]:', (z<20).sum() - (z<10).sum())

    # show distribution
    plt.title(split)
    plt.xlabel('x-value')
    plt.ylabel('z-label')
    plt.scatter(x_list, z_list, s=1)
    plt.savefig('./bev.png', dpi=300)


if __name__ == '__main__':
    get_kitti_bev_distribution()