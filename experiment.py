import os

import albumentations as A
import nntools.dataset as D
import nntools.tracker.metrics as NNmetrics
import numpy as np
import torch
import torch.distributed as dist
from nntools.dataset.image_tools import normalize
from nntools.experiment import Experiment
from nntools.utils import reduce_tensor, create_folder
from networks.networks import get_network
from utils import to_rgb, crop_fundus, quick_resize

from nntools.tracker import Log, Tracker, log_params, log_metrics, log_artifact
import deit_loss

class OCTClassification(Experiment):
    def __init__(self, config, run_id=None):
        super(OCTClassification, self).__init__(config, run_id=run_id)

        """
        Create network
        """
        architecture = self.config['Network']['architecture']
        img_size = self.config['Dataset']['shape']
        network = get_network(self.config['Network'], img_size)
        if hasattr(network, 'no_weight_decay'):
            remove_weight_decay = network.no_weight_decay()
        else:
            remove_weight_decay = {}
        network = self.set_model(network)

        params = []
        no_weight_params = []
        for name, tensor in network.named_parameters():
            if name in remove_weight_decay:
                no_weight_params.append(tensor)
            else:
                params.append(tensor)

        params_group = [{'params': params},
                        {'params': no_weight_params,
                         'weight_decay': 0}]
        network.set_params_group(params_group)

        """
        Data augmentation 
        """
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=10, p=0.5),
        ])
        composer = D.Composition(**self.config['Preprocessing'])
        composer << transform << to_rgb << normalize

        """
        Create datasets
        """
        if 'deit' in architecture:
            from datasets.deit_dataset import DeitDataset
            table_path = 'datasets/train_predictions_log.csv'
            dataset = DeitDataset(table_path, **self.config['Dataset'])
        else:
            dataset = D.ClassificationDataset(**self.config['Dataset'])
        dataset.set_composition(composer)
        valid_len = self.config['Validation']['size']
        train_len = len(dataset) - valid_len
        train_dataset, valid_dataset = D.random_split(dataset, [train_len, valid_len],
                                                      generator=torch.Generator().manual_seed(self.seed))

        test_config = self.config['Dataset'].copy()
        test_config['img_url'] = self.config['Test']['img_url']

        test_dataset = D.ClassificationDataset(**test_config)
        composer = D.Composition(**self.config['Preprocessing'])

        # composer << crop_fundus << quick_resize
        composer << to_rgb << normalize
        test_dataset.set_composition(composer)

        self.set_train_dataset(train_dataset)
        self.set_valid_dataset(valid_dataset)
        self.set_test_dataset(test_dataset)

        """
        Define optimizers
        """
        self.set_optimizer(**self.config['Optimizer'])
        self.set_scheduler(**self.config['Learning_rate_scheduler'])


        """
        Set the end and validate functions
        """
        self.log = {'file': [],
                    'prediction': [],
                    'probas': [],
                    'groundtruth': []}

    def start(self):
        """
       Start the run
       """
        self.start_run()
        log_artifact(self.tracker, os.path.realpath(__file__))
        super(OCTClassification, self).start()

    def validate(self, model, iteration, rank=0, loss_function=None):
        model.network.training = False
        valid_loader, valid_sampler = self.get_dataloader(self.validation_dataset, shuffle=False)
        confMat = torch.zeros(self.n_classes, self.n_classes).cuda(self.get_gpu_from_rank(rank))
        for batch in valid_loader:
            img = batch[0].cuda(self.get_gpu_from_rank(rank))
            gt = batch[1].cuda(self.get_gpu_from_rank(rank))
            pred = torch.argmax(model(img), 1)
            confMat += NNmetrics.confusion_matrix(pred, gt, num_classes=self.n_classes)
        if self.multi_gpu:
            confMat = reduce_tensor(confMat, self.world_size, mode='sum')

        confMat = NNmetrics.filter_index_cm(confMat, self.ignore_index)
        mIoU = NNmetrics.mIoU_cm(confMat)
        if self.is_main_process(rank):
            stats = NNmetrics.report_cm(confMat)
            stats['mIoU'] = mIoU
            log_metrics(self.tracker, step=iteration, **stats)
            if self.tracked_metric is None:
                self.tracked_metric = mIoU
            else:
                if mIoU > self.tracked_metric:
                    self.tracked_metric = mIoU
                    filename = ('best_valid_iteration_%i_mIoU_%.3f' % (iteration, mIoU)).replace('.', '')
                    self.save_model(model, filename=filename)
        model.network.training = True
        return mIoU

    def end(self, model, rank, save_proba=False):
        model.training = False
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.get_gpu_from_rank(rank)}
        model.load(self.tracker.network_savepoint, load_most_recent=True, map_location=map_location, strict=True)
        model.eval()
        self.test_dataset.return_indices = True
        test_loader, test_sampler = self.get_dataloader(self.test_dataset, shuffle=False, batch_size=1)
        confMat = torch.zeros(self.n_classes, self.n_classes).cuda(self.get_gpu_from_rank(rank))
        prediction_savepoint = self.tracker.prediction_savepoint
        if save_proba:
            prediction_savepoint = os.path.join(prediction_savepoint, 'labels/')
            if self.is_main_process(rank):
                create_folder(prediction_savepoint)
            probas_savepoint = os.path.join(prediction_savepoint, 'probas/')
            probas_savepoints = []
            for i in range(self.n_classes):
                probas_savepoints.append(os.path.join(probas_savepoint, 'class_%i/' % i))
                if self.is_main_process(rank):
                    create_folder(probas_savepoints[-1])
            if self.multi_gpu:
                dist.barrier()

        with torch.no_grad():
            for batch in test_loader:
                img = batch[0].cuda(self.get_gpu_from_rank(rank))
                gt = batch[1].cuda(self.get_gpu_from_rank(rank))
                indices = np.asarray(batch[2])
                filename = self.test_dataset.filename(indices)

                probas = model(img)
                preds = torch.argmax(probas, 1)
                confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes)
                probas = torch.softmax(probas, 1)

                for i in range(preds.shape[0]):
                    f = filename[i]
                    self.log['file'].append(f)
                    self.log['prediction'].append(preds[i].cpu().numpy())
                    self.log['probas'].append(probas[i].cpu().numpy())
                    self.log['groundtruth'].append(gt[i].cpu().numpy())

            if self.multi_gpu:
                confMat = reduce_tensor(confMat, self.world_size, mode='sum')
                dist.barrier()
            if self.is_main_process(rank):

                mIoU = NNmetrics.mIoU_cm(confMat)
                stats = NNmetrics.report_cm(confMat)
                stats['mIoU'] = mIoU
                test_scores = {}
                for k in stats:
                    test_scores['test_%s' % k] = stats[k]
                log_metrics(self.tracker, step=0, **test_scores)
                confMat = confMat.cpu().numpy()
                np.save(os.path.join(self.tracker.prediction_savepoint, 'test_confMat.npy'), confMat)
                log_artifact(self.tracker, os.path.join(self.tracker.prediction_savepoint, 'test_confMat.npy'))
                import pandas as pd
                df = pd.DataFrame.from_dict(self.log)
                df.to_csv(os.path.join(self.tracker.prediction_savepoint, 'test_predictions_log.csv'))
                log_artifact(self.tracker, os.path.join(self.tracker.prediction_savepoint, 'test_predictions_log.csv'))
