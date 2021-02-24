from nntools.utils import Config
from experiment import OCTClassification

if __name__ == '__main__':
    config_path = 'configs/config.yaml'
    config = Config(config_path)
    config['Manager']['run'] = 'vit_deit_base_distilled_patch16_384'
    config['Network']['architecture'] = config['Manager']['run']

    experiment = OCTClassification(config)
    experiment.start()
