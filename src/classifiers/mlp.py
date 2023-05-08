from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Metric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import FBetaScore

from composer import Trainer, ComposerModel
from composer.metrics import CrossEntropy
from composer.loggers import WandBLogger

from icecream import ic


class MLPModel(ComposerModel):
    """Implement MLP Classifier"""
    def __init__(self, embedding_size, h1=64, h2=32, num_classes=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # define model
        self.model = nn.Sequential(OrderedDict([
            ('input', nn.Linear(embedding_size, h1)),
            ('sig1', nn.Sigmoid()),
            ('hidden', nn.Linear(h1, h2)),
            ('sig2', nn.Sigmoid()),
            ('output', nn.Linear(h2, self.num_classes))
        ]))
        
        # Metrics for training
        self.train_metrics =  MetricCollection([
            Accuracy(num_classes=self.num_classes, average='micro'),
            FBetaScore(self.num_classes, mdmc_average='global')
            ])

        # Metrics for validation
        self.val_metrics = MetricCollection([
            CrossEntropy(),
            Accuracy(num_classes=self.num_classes, average='micro'), 
            FBetaScore(self.num_classes, mdmc_average='global')
            ])
 
    def forward(self, batch):
        inputs, _ = batch
        return self.model(inputs)
    
    def loss(self, outputs, batch):
        # pass batches and `forward` outputs to the loss
        _, targets = batch
        return F.cross_entropy(outputs, targets)
    
    def get_metrics(self, is_train=False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric: Metric) -> None:
        _, targets = batch
        metric.update(outputs, targets)
        

class MLP:
    
    def __init__(self, batch_size=64, duration='100ep', h1=64, h2=32, seed=42):
        self.num_classes = -1
        self.batch_size = batch_size
        self.duration = duration
        self.h1 = h1
        self.h2 = h2
        self.seed = seed
        
        self.trainer = None
        self.model = None
    
    def fit(self, X_train, y_train):
        
        self.num_classes = np.max(y_train) + 1
        
        # torch dataloader
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train)
            )
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        
        # model
        embedding_size = X_train.shape[1]
        model = MLPModel(embedding_size, h1=self.h1, h2=self.h2, num_classes=self.num_classes)
        self.model = model
        
        # logger
        wandb_logger = WandBLogger(project='IBD Classifier')

        # trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            optimizers=torch.optim.Adam(model.parameters(), lr=0.01),
            max_duration=self.duration,
            device='gpu',
            progress_bar=False,
            load_progress_bar=False,
            seed=self.seed,
            
        )
        self.trainer = trainer
        
        # fit
        self.trainer.fit()
        
        
    def predict(self, X_test, y_test=None):
        
        # torch dataloader
        if y_test is None:
            y_test = np.zeros_like(X_test)
        eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test.astype(np.float32)), 
                                                      torch.from_numpy(y_test.astype(np.float32)))       
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        # run predict
        y_pred = self.trainer.predict(eval_dataloader)
        _, y_pred = torch.vstack(tuple(y_pred)).max(1)
        return y_pred