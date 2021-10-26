import torch
import hydra
from omegaconf import DictConfig

from src.dataset import get_train_dataloader, get_test_dataloader
from src.models import LinearNet
from src.running import Runner
from src.tensorboard import TensorboardExperiment
from src.tracking import Stage
from src.utils import generate_tensorboard_experiment_directory

# Hyperparameters
hparams = {
    'EPOCHS': 20,
    'LR': 5e-5,
    'OPTIMIZER': 'Adam',
    'BATCH_SIZE': 128
}


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Data
    train_loader = get_train_dataloader(batch_size=cfg.hparams.batch_size)
    test_loader = get_test_dataloader(batch_size=cfg.hparams.batch_size)

    # Model and Optimizer
    model = LinearNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hparams.lr)

    # Create the runners
    test_runner = Runner(test_loader, model)
    train_runner = Runner(train_loader, model, optimizer)

    # Experiment Trackers
    log_dir = generate_tensorboard_experiment_directory(root='./runs')
    experiment = TensorboardExperiment(log_dir=log_dir)

    for epoch in range(cfg.hparams.epochs):
        experiment.set_stage(Stage.TRAIN)
        train_runner.run("Train batches", experiment)

        # Log Training Epoch Metrics
        experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch)

        experiment.set_stage(Stage.VAL)
        test_runner.run("Validation batches", experiment)

        # Log Training Epoch Metrics
        experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch)
        experiment.add_epoch_confusion_matrix(
            test_runner.y_true_batches, test_runner.y_pred_batches, epoch
        )

        # Compute Average Epoch Metrics
        summary = ', '.join([
            f"[Epoch: {epoch + 1}/{cfg.hparams.epochs}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ])
        print('\n' + summary + '\n')

        test_runner.reset()
        train_runner.reset()

    experiment.flush()


if __name__ == '__main__':
    main()
