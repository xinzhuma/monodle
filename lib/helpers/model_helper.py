from lib.models.centernet3d import CenterNet3D


def build_model(cfg):
    if cfg['type'] == 'centernet3d':
        return CenterNet3D(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'])
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


