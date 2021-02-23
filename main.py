from nntools.utils import Config

from experiment import OCTClassification
from fundus_experiment import FundusClassification

if __name__ == '__main__':
    config_path = 'config.yaml'
    config = Config(config_path)
    config['Manager']['run'] = 'ResNet152'
    config['Network']['architecture'] = config['Manager']['run']

    experiment = OCTClassification(config)
    experiment.start()
