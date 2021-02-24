from torchvision.models.resnet import ResNet, model_urls, Bottleneck
from torch.hub import load_state_dict_from_url

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')

        model.load_state_dict(state_dict, strict=False)
    return model


def get_network(config):
    if config['architecture'] == 'Wide ResNet-101-2':
        network = _resnet('wide_resnet101_2',
                          Bottleneck, [3, 4, 23, 3],
                          pretrained=config['pretrained'],
                          progress=True,
                          num_classes=config['n_classes'], width_per_group=64 * 2)

    if config['architecture'] == 'ResNet152':
        network = _resnet('resnet152',
                          Bottleneck, [3, 8, 36, 3],
                          pretrained=config['pretrained'],
                          progress=True,
                          num_classes=config['n_classes'])

    if config['architecture'] == 'vit_base_patch32_384':
        from timm.models.vision_transformer import vit_base_patch32_384
        network = vit_base_patch32_384(pretrained=config['pretrained'], num_classes=config['n_classes'])

    if config['architecture'] == 'vit_deit_base_distilled_patch16_384':
        from timm.models.vision_transformer import vit_deit_base_distilled_patch16_384
        network = vit_deit_base_distilled_patch16_384(pretrained=False, num_classes=config['n_classes'])
        if config['pretrained']:
            state_dict = load_state_dict_from_url('https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
                                                  progress=True, map_location='cpu')['model']
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            del state_dict['head_dist' + '.weight']
            del state_dict['head_dist' + '.bias']
            network.load_state_dict(state_dict, strict=False)

    if config['architecture'] == 'vit_large_patch16_384':
        from timm.models.vision_transformer import vit_large_patch16_384
        network = vit_large_patch16_384(pretrained=config['pretrained'], num_classes=config['n_classes'])


    if config['architecture'] == 'OpticNet':
            from networks.opticNet import OpticNet
            network = OpticNet(n_classes=config['n_classes'], n_channels=3)

    return network
