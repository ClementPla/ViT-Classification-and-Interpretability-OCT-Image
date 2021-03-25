from nntools.utils import Config
from experiment import OCTClassification

if __name__ == '__main__':
    config_path = 'configs/config.yaml'
    config = Config(config_path)
    config['Manager']['run'] = 'vit_base_patch16_384'
    config['Network']['architecture'] = config['Manager']['run']

    experiment = OCTClassification(config, '155192df68344024bb36d059d1d35229')
    experiment.run_training = False
    experiment.start()
