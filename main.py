from nntools.utils import Config
from experiment import OCTClassification
from fundus_experiment import FundusClassification

if __name__ == '__main__':
    config_path = 'configs/config_fundus.yaml'
    config = Config(config_path)

    config['Manager']['run'] = 'ResNet152'
    config['Network']['architecture'] = config['Manager']['run']

    experiment = FundusClassification(config)
    experiment.start()
