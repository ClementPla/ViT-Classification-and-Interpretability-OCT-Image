import torch
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, model_urls, Bottleneck


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')

        model.load_state_dict(state_dict, strict=False)
    return model


MODELS_URLS = {
    'T2T-ViT-14': 'https://public.sn.files.1drv.com/y4mOgjplffBRfwVd5vXUcQahto-3pi68kx00md-ZMbw1RTzHag2ncAmRHNKHmkAJyUQlgoEhCfdnhrekNFE2a-dk7-7-mwkCl2FOUZoCPMn0dPwEApbl5wGUt7pxi7VuSvm_-HkavshZWlx23bvU2OuKSx6QhO9BRWqnEPmBj-dEzKinSQC7WPTfBBMGqeFrpLSfNqWdTbJ5xhFJefZ9EJMbZT3if2ijD2VFg0nJZiL5xU?access_token=EwD4Aq1DBAAUmcDj0azQ5tf1lkBfAvHLBzXl5ugAAZpOEZrZsspk9vu/ez7LhqgNy0FWAtEeQvjHnrjwKzfGTFnNJ7Bx0g2z/TYiGiNbWvcI9zwgGl6pZiUv7P4HcHjvYw8UbRv7hGKfWy%2baqjLjZ/bEIIZwwytaiFvPotCnIzvWA3OPwp1NbpnhS8ESUezZa1eI82JGKKqfzo8IaSQKUK0A5TiFEwRub44U3tttNjAm6WCwN%2bYcoWOrvAnHdc26QS2dEEAWy4vPcyU3nHri33vlXmw4xAxrnlvQWUsegTL%2bSbUrmI0tiOZhMhxTJIQMD7yTbdbv6lj9E99VU4aML7xm/tlHDjLGYxuhTlZgYeP9h7uHbgC80mP6JJ3okDcDZgAACPRKzois%2bbEFyAFKH8GepcXYlfe5i7Lg3Fi35r98LSFfcW0qBiFjCqcR7U2tKZJCZsCAS8hqtEvPSEjaiCpEIHfrVmj2aGoltaKNCISG5G8LINo6MdLuTlBy5b7s%2by4oLI66bcYxiJBdlVNHYKbzXvKMbTdHKXTcI/jM6Nqm0QcovKeRSmmd/2EogQB5MlVCRuvoSeKMbfZmWDCzznv9WDZwXlF%2bDW4M34EmM44nk%2b6z8PMrhr0AHlU0sJkjhCC7JJBEXo0WvstpFDe41PEI8nlAvIT8kDHVaXTdWl90wnGkNh73%2bSr8T%2b0GzJeU%2bb%2bKLJQV7LMWyDlEDT0HufAmJKRgUk6RTgAmjFPL37yqyabjxFcJbOB4RawQazhflgtxY8HeSJnSTcO4pE1TM4t7Inw/J3xeCPEMu/BuKf8EPgG497lWgqACQn5ZsYREJSQG0ZXkRULprmBeb4swLIIdHv85djnaLK04588zciHdgl3%2btVmlV0d%2b1%2bQTMSDrRSbpV7qYfANq/LX/m1uaA9QyxrJtzucLA7iqVbFdRBE5JPu%2bHmE7bptHOkcYdRxhLz36PoaZRJ7Xh2BcUVUk%2bInuh%2bAoeJwKkgcTgkKPmmCVfbxOlGYDAg%3d%3d',
    'T2T-ViT_t-24': 'https://public.sn.files.1drv.com/y4m4dbjHdaZCr5Jlb1oiTPOv0XiPd2NhUnkLalrD5pts_RukufaJD0Dki0Dm01bQ13jlhrMrGJjk2dn24Uoz2IIJu6aO_dss46UqAbmUp8QQd2u4bdaSxvzT-pTuhah4MKhs7KPEfknYiOFK9AA_ixK6hzAFokiqDppbkku2506HAtwl61wmtHu3hH1hwEevTxW_aEthRhBndIerzy8e3qNzoAPjiWqad-IBQFik6KnfS8?access_token=EwD4Aq1DBAAUmcDj0azQ5tf1lkBfAvHLBzXl5ugAAZpOEZrZsspk9vu%2Fez7LhqgNy0FWAtEeQvjHnrjwKzfGTFnNJ7Bx0g2z%2FTYiGiNbWvcI9zwgGl6pZiUv7P4HcHjvYw8UbRv7hGKfWy%2BaqjLjZ%2FbEIIZwwytaiFvPotCnIzvWA3OPwp1NbpnhS8ESUezZa1eI82JGKKqfzo8IaSQKUK0A5TiFEwRub44U3tttNjAm6WCwN%2BYcoWOrvAnHdc26QS2dEEAWy4vPcyU3nHri33vlXmw4xAxrnlvQWUsegTL%2BSbUrmI0tiOZhMhxTJIQMD7yTbdbv6lj9E99VU4aML7xm%2FtlHDjLGYxuhTlZgYeP9h7uHbgC80mP6JJ3okDcDZgAACPRKzois%2BbEFyAFKH8GepcXYlfe5i7Lg3Fi35r98LSFfcW0qBiFjCqcR7U2tKZJCZsCAS8hqtEvPSEjaiCpEIHfrVmj2aGoltaKNCISG5G8LINo6MdLuTlBy5b7s%2By4oLI66bcYxiJBdlVNHYKbzXvKMbTdHKXTcI%2FjM6Nqm0QcovKeRSmmd%2F2EogQB5MlVCRuvoSeKMbfZmWDCzznv9WDZwXlF%2BDW4M34EmM44nk%2B6z8PMrhr0AHlU0sJkjhCC7JJBEXo0WvstpFDe41PEI8nlAvIT8kDHVaXTdWl90wnGkNh73%2BSr8T%2B0GzJeU%2Bb%2BKLJQV7LMWyDlEDT0HufAmJKRgUk6RTgAmjFPL37yqyabjxFcJbOB4RawQazhflgtxY8HeSJnSTcO4pE1TM4t7Inw%2FJ3xeCPEMu%2FBuKf8EPgG497lWgqACQn5ZsYREJSQG0ZXkRULprmBeb4swLIIdHv85djnaLK04588zciHdgl3%2BtVmlV0d%2B1%2BQTMSDrRSbpV7qYfANq%2FLX%2Fm1uaA9QyxrJtzucLA7iqVbFdRBE5JPu%2BHmE7bptHOkcYdRxhLz36PoaZRJ7Xh2BcUVUk%2BInuh%2BAoeJwKkgcTgkKPmmCVfbxOlGYDAg%3D%3D',
    'T2T-ViT-19': 'https://public.sn.files.1drv.com/y4mHfHSXs9faVkDogD0osdFaSumfrvWILYkokKTdja8phm5dnEm0mhmU-RJhYkkr3PQ7RKKBWP5be7md5CV5iEBv40gnOkBiaeGcPW8rKYKvG4a6wrL4UB8pd0R2hA3dpcrYD2jQb26dp4KTiDu9UOsIIDTUwCGdISPIYTgwecTeUOyFYlHlGmVOQJGgprC512oAcm_1ui_5URUKsUpvLlb_owe-KTi68Ufrfa7dQ9IwVw?access_token=EwD4Aq1DBAAUmcDj0azQ5tf1lkBfAvHLBzXl5ugAAZpOEZrZsspk9vu/ez7LhqgNy0FWAtEeQvjHnrjwKzfGTFnNJ7Bx0g2z/TYiGiNbWvcI9zwgGl6pZiUv7P4HcHjvYw8UbRv7hGKfWy%2baqjLjZ/bEIIZwwytaiFvPotCnIzvWA3OPwp1NbpnhS8ESUezZa1eI82JGKKqfzo8IaSQKUK0A5TiFEwRub44U3tttNjAm6WCwN%2bYcoWOrvAnHdc26QS2dEEAWy4vPcyU3nHri33vlXmw4xAxrnlvQWUsegTL%2bSbUrmI0tiOZhMhxTJIQMD7yTbdbv6lj9E99VU4aML7xm/tlHDjLGYxuhTlZgYeP9h7uHbgC80mP6JJ3okDcDZgAACPRKzois%2bbEFyAFKH8GepcXYlfe5i7Lg3Fi35r98LSFfcW0qBiFjCqcR7U2tKZJCZsCAS8hqtEvPSEjaiCpEIHfrVmj2aGoltaKNCISG5G8LINo6MdLuTlBy5b7s%2by4oLI66bcYxiJBdlVNHYKbzXvKMbTdHKXTcI/jM6Nqm0QcovKeRSmmd/2EogQB5MlVCRuvoSeKMbfZmWDCzznv9WDZwXlF%2bDW4M34EmM44nk%2b6z8PMrhr0AHlU0sJkjhCC7JJBEXo0WvstpFDe41PEI8nlAvIT8kDHVaXTdWl90wnGkNh73%2bSr8T%2b0GzJeU%2bb%2bKLJQV7LMWyDlEDT0HufAmJKRgUk6RTgAmjFPL37yqyabjxFcJbOB4RawQazhflgtxY8HeSJnSTcO4pE1TM4t7Inw/J3xeCPEMu/BuKf8EPgG497lWgqACQn5ZsYREJSQG0ZXkRULprmBeb4swLIIdHv85djnaLK04588zciHdgl3%2btVmlV0d%2b1%2bQTMSDrRSbpV7qYfANq/LX/m1uaA9QyxrJtzucLA7iqVbFdRBE5JPu%2bHmE7bptHOkcYdRxhLz36PoaZRJ7Xh2BcUVUk%2bInuh%2bAoeJwKkgcTgkKPmmCVfbxOlGYDAg%3d%3d'
}


# Models are rehosted on a personal OneDrive, allowing direct retrieval


def get_network(config, img_size=None):
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
        network = vit_deit_base_distilled_patch16_384(pretrained=False, num_classes=config['n_classes'], img_size=img_size)
        if config['pretrained']:
            state_dict = \
                load_state_dict_from_url(
                    'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
                    progress=True, map_location='cpu')['model']
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            del state_dict['head_dist' + '.weight']
            del state_dict['head_dist' + '.bias']
            if img_size != 384:
                del state_dict['pos_embed']
            network.load_state_dict(state_dict, strict=False)

    if config['architecture'] == 'vit_large_patch16_384':
        from timm.models.vision_transformer import vit_large_patch16_384
        network = vit_large_patch16_384(pretrained=config['pretrained'], num_classes=config['n_classes'], img_size=img_size, drop_rate=0.1)
    if config['architecture'] == 'vit_base_patch16_384':
        from timm.models.vision_transformer import vit_base_patch16_384
        network = vit_base_patch16_384(pretrained=config['pretrained'], num_classes=config['n_classes'])
    
    if config['architecture'] == 'T2T-ViT-14':
        from .T2T.models import T2t_vit_14
        network = T2t_vit_14(num_classes=config['n_classes'], img_size=img_size)
        if config['pretrained']:
            url_pretrained = MODELS_URLS['T2T-ViT-14']
            state_dict = load_state_dict_from_url(url_pretrained, map_location='cpu')['state_dict_ema']
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            if img_size != 224:
                del state_dict['pos_embed']
            network.load_state_dict(state_dict, strict=False)
    if config['architecture'] == 'T2T-ViT_t-24':
        from .T2T.models import T2t_vit_t_24
        network = T2t_vit_t_24(num_classes=config['n_classes'], img_size=img_size)
        if config['pretrained']:
            url_pretrained = MODELS_URLS['T2T-ViT_t-24']
            state_dict = load_state_dict_from_url(url_pretrained, map_location='cpu')['state_dict_ema']
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            if img_size != 224:
                del state_dict['pos_embed']
            network.load_state_dict(state_dict, strict=False)

    if config['architecture'] == 'T2T-ViT-19':
        from .T2T.models import T2t_vit_19
        network = T2t_vit_19(num_classes=config['n_classes'], img_size=img_size)
        if config['pretrained']:
            url_pretrained = MODELS_URLS['T2T-ViT-19']
            state_dict = load_state_dict_from_url(url_pretrained, map_location='cpu')['state_dict_ema']
            del state_dict['head' + '.weight']
            del state_dict['head' + '.bias']
            if img_size != 224:
                del state_dict['pos_embed']
            network.load_state_dict(state_dict, strict=False)

    if config['architecture'] == 'OpticNet':
        from networks.opticNet import OpticNet
        network = OpticNet(n_classes=config['n_classes'], n_channels=3)

    return network
