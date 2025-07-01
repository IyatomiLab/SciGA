import os
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import benchmark.models as models
import benchmark.models.classifier_loader as model_loader
import benchmark.datasets as datasets
import benchmark.metrics as metrics
import benchmark.utils as utils
from tap import Tap
from tqdm import tqdm
from datetime import datetime
import wandb


class Args(Tap):
    experiment_name: str = 'SciGA_intraGA_GA-BC'
    seed: int = 42
    device: int = 0
    dataset_json_dir: str = './SciGA_for_experiments/json/'
    dataset_figure_dir: str = './SciGA_for_experiments/figures/'
    save_checkpoint_dir: str = './benchmark/output/checkpoints/'

    model_type: str = 'ViT'
    model_name: str = None
    weight_decay: float = 1e-3
    epochs: int = 16
    num_workers: int = 4
    batch_size: int = 16
    accum_iter: int = 64
    learning_rate: float = 1e-6

    is_wandb: bool = False

    def configure(self):
        self.add_argument('--is_wandb', action='store_true', help='Use Weights and Biases for logging', required=False)


class Experiment():
    def __init__(
        self,
        model: models.BaseAbs2FigRetrieverForIntraGARecommendation,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        args: Args
    ) -> None:
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        print(f'\n🚀 Using device: {self.device}')

        self.experiment_name = args.experiment_name
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.accum_iter = args.accum_iter
        self.save_checkpoint_dir = args.save_checkpoint_dir
        self.is_wandb = args.is_wandb

    def train(self) -> tuple[dict[str, list[float]], dict[str, list[dict[str, float | tuple[float, float]]]]]:
        # Set training settings
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler(self.device.type)
        torch.backends.cudnn.benchmark = True

        loss_history = {'train': [], 'valid': []}
        score_history = {'train': [], 'valid': []}

        # Epoch loop
        for epoch in range(self.epochs):
            print(f'\n📈 Running Training... [{epoch+1}/{self.epochs}]')

            self.model.train()
            epoch_loss = 0.0
            epoch_result = []
            epoch_score = {}

            # Training
            for i, (paper_id, research_fields, figure_id, GT_figure_ids, figure, label) in enumerate(tqdm(self.train_dataloader, desc='   train', ncols=80)):
                figure = figure.to(self.device)
                label = label.to(self.device)

                # Forward
                with torch.amp.autocast(self.device.type):
                    output: models.GAClassifierOutput = self.model(
                        figure,
                        label,
                    )
                    loss = output.loss
                    probs = output.prob

                # Backward
                self.scaler.scale(loss / self.accum_iter).backward()
                if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_dataloader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Save loss and result
                epoch_loss += loss.item()
                for j, id in enumerate(paper_id):
                    epoch_result.append({
                        'paper_id': id,
                        'figure_id': figure_id[j],
                        'prob': probs[j][1].item(),
                        'GT_figure_ids': GT_figure_ids[j],
                    })

            # Record average loss
            epoch_loss = epoch_loss / len(self.train_dataloader)
            loss_history['train'].append(epoch_loss)

            # Record score
            epoch_result = pd.DataFrame(epoch_result)
            recall, mrr, car, car_above05 = metrics.evaluate_intraGA_recommendation_metrics(epoch_result)
            epoch_score = {
                'R@1': recall['1'],
                'R@2': recall['2'],
                'R@3': recall['3'],
                'MRR': mrr,
                'CAR@5': car['5'],
                'CAR@5_above_0.5': car_above05['5'],
            }
            score_history['train'].append(epoch_score)

            # Log training results to Weights and Biases
            if self.is_wandb:
                wandb.log({
                    'train_loss': loss_history['train'][-1],
                    'train_R@1': score_history['train'][-1]['R@1'],
                    'train_R@2': score_history['train'][-1]['R@2'],
                    'train_R@3': score_history['train'][-1]['R@3'],
                    'train_MRR': score_history['train'][-1]['MRR'],
                    'train_CAR@5': score_history['train'][-1]['CAR@5'],
                    'train_CAR@5_above_0.5': score_history['train'][-1]['CAR@5_above_0.5'],
                }, step=epoch)

            # Validation
            epoch_loss, epoch_score = self.test(self.valid_dataloader, desc='valid')
            loss_history['valid'].append(epoch_loss)
            score_history['valid'].append(epoch_score)

            # Log validation results to Weights and Biases
            if self.is_wandb:
                wandb.log({
                    'valid_loss': loss_history['valid'][-1],
                    'valid_R@1': score_history['valid'][-1]['R@1'],
                    'valid_R@2': score_history['valid'][-1]['R@2'],
                    'valid_R@3': score_history['valid'][-1]['R@3'],
                    'valid_MRR': score_history['valid'][-1]['MRR'],
                    'valid_CAR@5': score_history['valid'][-1]['CAR@5'],
                    'valid_CAR@5_above_0.5': score_history['valid'][-1]['CAR@5_above_0.5'],
                }, step=epoch)

            utils.print_epoch_scores(
                phases=['train', 'valid'],
                losses=[loss_history['train'][-1], loss_history['valid'][-1]],
                scores=[score_history['train'][-1], score_history['valid'][-1]]
            )

            self.scheduler.step()

        # Save checkpoint
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = f'{self.save_checkpoint_dir}{self.experiment_name}_{timestamp}.pt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.get_backbone().state_dict(), path)
        print(f'\n💾 Saved model checkpoint to: \'{path}\'')

        return loss_history, score_history

    @torch.inference_mode()
    def test(self, dataloader: DataLoader, desc: str) -> tuple[float, dict[str, float | tuple[float, float]]]:
        self.model.eval()
        epoch_loss = 0.0
        epoch_result = []

        for i, (paper_id, research_fields, figure_id, GT_figure_ids, figure, label) in enumerate(tqdm(dataloader, desc='   ' + desc, ncols=80)):
            figure = figure.to(self.device)
            label = label.to(self.device)

            # Forward
            with torch.amp.autocast(self.device.type):
                output: models.GAClassifierOutput = self.model(
                    figure,
                    label,
                )
                loss = output.loss
                probs = output.prob

            # Save loss and result
            epoch_loss += loss.item()
            for j, id in enumerate(paper_id):
                epoch_result.append({
                    'paper_id': id,
                    'figure_id': figure_id[j],
                    'prob': probs[j][1].item(),
                    'GT_figure_ids': GT_figure_ids[j],
                })

        # Record average loss
        epoch_loss = epoch_loss / len(dataloader)

        # Record score
        epoch_result = pd.DataFrame(epoch_result)
        recall, mrr, car, car_above05 = metrics.evaluate_intraGA_recommendation_metrics(epoch_result)
        epoch_score = {
            'R@1': recall['1'],
            'R@2': recall['2'],
            'R@3': recall['3'],
            'MRR': mrr,
            'CAR@5': car['5'],
            'CAR@5_above_0.5': car_above05['5'],
        }

        return epoch_loss, epoch_score


def main(args: Args) -> None:
    is_wandb = args.is_wandb
    experiment_name = args.experiment_name
    seed = args.seed
    dataset_json_dir = args.dataset_json_dir
    dataset_figure_dir = args.dataset_figure_dir
    model_type = args.model_type
    model_name = args.model_name or None
    batch_size = args.batch_size
    num_workers = args.num_workers

    print(f'\n🔍 Experiment Name: {experiment_name}')

    # Set up Weights and Biases
    if is_wandb:
        wandb.login()
        wandb.init(project='SciGA', name=f'{experiment_name}', config=args.__dict__)

    # Set seed
    utils.set_seed(seed)

    # Load data
    print(f'\n📦 Loading dataset from: \'{dataset_json_dir}\'...')
    train = datasets.load_and_prepare_split(
        split_name='train',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
        is_classification=True,
    )
    valid = datasets.load_and_prepare_split(
        split_name='valid',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
        is_classification=True,
    )
    test = datasets.load_and_prepare_split(
        split_name='test',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
        is_classification=True,
    )
    print(f'   train: {len(train)}, valid: {len(valid)}, test: {len(test)}')

    # Load model
    print(f'\n📦 Loading model({model_type})...')
    num_GAs = (train['label'] == 1).sum()
    num_figures = (train['label'] == 0).sum()
    num_images = [num_figures, num_GAs]
    loss_weight = torch.tensor([1.0 / num_images[i] for i in range(2)], dtype=torch.float32)

    model = model_loader.load_GA_classifier(
        model_type=model_type,
        model_name=model_name,
        loss_weight=loss_weight,
    )

    # Create Dataset
    train_dataset = datasets.GABinaryClassifierDataset(
        paper_id=train['paper_id'].tolist(),
        research_fields=train['research_fields'].tolist(),
        image_path=train['image_path'].tolist(),
        label=train['label'].tolist(),
        transform=model.preprocess_image,
        GT_figure_ids=train['GT_figure_ids'].tolist(),
    )
    valid_dataset = datasets.GABinaryClassifierDataset(
        paper_id=valid['paper_id'].tolist(),
        research_fields=valid['research_fields'].tolist(),
        image_path=valid['image_path'].tolist(),
        label=valid['label'].tolist(),
        transform=model.preprocess_image,
        GT_figure_ids=valid['GT_figure_ids'].tolist(),
    )
    test_dataset = datasets.GABinaryClassifierDataset(
        paper_id=test['paper_id'].tolist(),
        research_fields=test['research_fields'].tolist(),
        image_path=test['image_path'].tolist(),
        label=test['label'].tolist(),
        transform=model.preprocess_image,
        GT_figure_ids=test['GT_figure_ids'].tolist(),
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=valid_dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Run experiments
    experiment = Experiment(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        args=args
    )

    _, _ = experiment.train()

    print('\n🧪 Running Test Evaluation...')
    loss, score = experiment.test(test_dataloader, desc='test')
    utils.print_epoch_scores(
        phases=['test'],
        losses=[loss],
        scores=[score]
    )


if __name__ == '__main__':
    args = Args().parse_args()
    main(args)
