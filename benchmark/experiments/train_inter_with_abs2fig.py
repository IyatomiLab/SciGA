import os
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import benchmark.models as models
import benchmark.models.retriever_loader as model_loader
import benchmark.datasets as datasets
import benchmark.metrics as metrics
import benchmark.utils as utils
from tap import Tap
from tqdm import tqdm
from datetime import datetime
import wandb


class Args(Tap):
    experiment_name: str = 'SciGA_interGA_Abs2Fig'
    seed: int = 42
    device: int = 0
    dataset_json_dir: str = './SciGA_for_experiments/json/'
    dataset_figure_dir: str = './SciGA_for_experiments/figures/'
    save_checkpoint_dir: str = './benchmark/output/checkpoints/'
    save_cache_dir: str = './benchmark/output/caches/'

    model_type: str = 'CLIP'
    model_name: str = None
    model_config_path: str = None
    weight_decay: float = 1e-3
    epochs: int = 16
    num_workers: int = 4
    batch_size: int = 16
    accum_iter: int = 64
    learning_rate: float = 1e-6
    is_merge_caption: bool = False

    is_wandb: bool = False

    def configure(self):
        self.add_argument('--is_merge_caption', action='store_true', help='Merge Caption Embeddings into Figure Embeddings', required=False)
        self.add_argument('--is_wandb', action='store_true', help='Use Weights and Biases for logging', required=False)


class Experiment():
    def __init__(
        self,
        model: models.BaseAbs2FigRetrieverForInterGARecommendation,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        candidate_dataloader: DataLoader,
        SBERT_embeddings: dict[str, torch.Tensor],
        CLIP_embeddings: dict[str, torch.Tensor],
        DreamSim_embeddings: dict[str, torch.Tensor],
        args: Args
    ) -> None:
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = args.experiment_name
        self.is_merge_caption = args.is_merge_caption
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.candidate_dataloader = candidate_dataloader
        self.SBERT_embeddings = SBERT_embeddings
        self.CLIP_embeddings = CLIP_embeddings
        self.DreamSim_embeddings = DreamSim_embeddings
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
            for i, (paper_id, research_fields, abstract, GA, caption) in enumerate(tqdm(self.train_dataloader, desc='   train', ncols=80)):
                abstract = abstract.to(self.device)
                GA = GA.to(self.device)
                caption = caption.to(self.device) if self.is_merge_caption else None

                # Forward
                with torch.amp.autocast(self.device.type):
                    output: models.Abs2FigRetrieverOutputForInterGARecommendation = self.model(
                        abstract,
                        GA,
                        caption,
                    )
                    loss = output.inter_loss
                    sim_abs2GA = output.sim_abs2GA.detach().cpu()

                    probs, preds = sim_abs2GA.sort(dim=-1, descending=True)
                    probs = probs.squeeze(1)
                    preds = preds.squeeze(1)

                # Backward
                self.scaler.scale(loss / self.accum_iter).backward()
                if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_dataloader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Save loss and result
                epoch_loss += loss.item()
                for j, id in enumerate(paper_id):
                    for k, prob in zip(preds[j], probs[j]):
                        k = k.item()
                        prob = prob.item()
                        epoch_result.append({
                            'paper_id': id,
                            'research_fields': research_fields[j],
                            'retrieved_paper_id': paper_id[k],
                            'retrieved_research_fields': research_fields[k],
                            'prob': prob,
                        })

            # Record average loss
            epoch_loss = epoch_loss / len(self.train_dataloader)
            loss_history['train'].append(epoch_loss)

            # Record score
            epoch_result = pd.DataFrame(epoch_result)
            mean_field_precision, mean_abs2abs_SBERT, std_abs2abs_SBERT, mean_GA2GA_CLIPScore, std_GA2GA_CLIPScore, mean_GA2GA_DreamSim, std_GA2GA_DreamSim = metrics.evaluate_interGA_recommendation_metrics(
                epoch_result,
                SBERT_embeddings=self.SBERT_embeddings,
                CLIP_embeddings=self.CLIP_embeddings,
                DreamSim_embeddings=self.DreamSim_embeddings,
                k_for_field_precision=[1],
                k_for_abs2abs_SBERT=[1],
                k_for_GA2GA_CLIPScore=[1],
            )
            epoch_score = {
                'Field-P@1': mean_field_precision['1'],
                'Field-P@5': None,
                'Field-P@10': None,
                'Abs2Abs_SBERT@1': (mean_abs2abs_SBERT['1'], std_abs2abs_SBERT['1']),
                'Abs2Abs_SBERT@5': None,
                'Abs2Abs_SBERT@10': None,
                'GA2GA_CLIP-S@1': (mean_GA2GA_CLIPScore['1'], std_GA2GA_CLIPScore['1']),
                'GA2GA_CLIP-S@5': None,
                'GA2GA_CLIP-S@10': None,
                'GA2GA_DreamSim@1': (mean_GA2GA_DreamSim['1'], std_GA2GA_DreamSim['1']),
                'GA2GA_DreamSim@5': None,
                'GA2GA_DreamSim@10': None,
            }
            score_history['train'].append(epoch_score)

            # Log training results to Weights and Biases
            if self.is_wandb:
                wandb.log({
                    'train_loss': loss_history['train'][-1],
                    'train_Field-P@1': score_history['train'][-1]['Field-P@1'],
                    'train_Abs2Abs_SBERT@1_mean': score_history['train'][-1]['Abs2Abs_SBERT@1'][0],
                    'train_Abs2Abs_SBERT@1_std': score_history['train'][-1]['Abs2Abs_SBERT@1'][1],
                    'train_GA2GA_CLIP-S@1_mean': score_history['train'][-1]['GA2GA_CLIP-S@1'][0],
                    'train_GA2GA_CLIP-S@1_std': score_history['train'][-1]['GA2GA_CLIP-S@1'][1],
                    'train_GA2GA_DreamSim@1_mean': score_history['train'][-1]['GA2GA_DreamSim@1'][0],
                    'train_GA2GA_DreamSim@1_std': score_history['train'][-1]['GA2GA_DreamSim@1'][1],
                }, step=epoch)

            # Validation
            epoch_loss, epoch_score = self.test(self.valid_dataloader, desc='valid')
            loss_history['valid'].append(epoch_loss)
            score_history['valid'].append(epoch_score)

            # Log validation results to Weights and Biases
            if self.is_wandb:
                wandb.log({
                    'valid_loss': loss_history['valid'][-1],
                    'valid_Field-P@5': score_history['valid'][-1]['Field-P@5'],
                    'valid_Field-P@10': score_history['valid'][-1]['Field-P@10'],
                    'valid_Abs2Abs_SBERT@5_mean': score_history['valid'][-1]['Abs2Abs_SBERT@5'][0],
                    'valid_Abs2Abs_SBERT@5_std': score_history['valid'][-1]['Abs2Abs_SBERT@5'][1],
                    'valid_Abs2Abs_SBERT@10_mean': score_history['valid'][-1]['Abs2Abs_SBERT@10'][0],
                    'valid_Abs2Abs_SBERT@10_std': score_history['valid'][-1]['Abs2Abs_SBERT@10'][1],
                    'valid_GA2GA_CLIP-S@5_mean': score_history['valid'][-1]['GA2GA_CLIP-S@5'][0],
                    'valid_GA2GA_CLIP-S@5_std': score_history['valid'][-1]['GA2GA_CLIP-S@5'][1],
                    'valid_GA2GA_CLIP-S@10_mean': score_history['valid'][-1]['GA2GA_CLIP-S@10'][0],
                    'valid_GA2GA_CLIP-S@10_std': score_history['valid'][-1]['GA2GA_CLIP-S@10'][1],
                    'valid_GA2GA_DreamSim@5_mean': score_history['valid'][-1]['GA2GA_DreamSim@5'][0],
                    'valid_GA2GA_DreamSim@5_std': score_history['valid'][-1]['GA2GA_DreamSim@5'][1],
                    'valid_GA2GA_DreamSim@10_mean': score_history['valid'][-1]['GA2GA_DreamSim@10'][0],
                    'valid_GA2GA_DreamSim@10_std': score_history['valid'][-1]['GA2GA_DreamSim@10'][1],
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

        # Precompute embeddings for candidate GAs (from the training set)
        candidate_GA_embed = []
        candidate_paper_id = []
        candidate_research_fields = []
        for i, (paper_id, research_fields, abstract, GA, caption) in enumerate(tqdm(self.candidate_dataloader, desc='   candidate', ncols=80)):
            abstract = abstract.to(self.device)
            GA = GA.to(self.device)
            caption = caption.to(self.device) if self.is_merge_caption else None

            with torch.amp.autocast(self.device.type):
                output: models.Abs2FigRetrieverOutputForInterGARecommendation = self.model(
                    abstract,
                    GA,
                    caption,
                )
                candidate_GA_embed.append(output.GA_embed)
                candidate_paper_id.extend(paper_id)
                candidate_research_fields.extend(research_fields)

        candidate_GA_embed = torch.cat(candidate_GA_embed, dim=0)

        # Perform retrieval
        for i, (paper_id, research_fields, abstract, GA, caption) in enumerate(tqdm(dataloader, desc='   ' + desc, ncols=80)):
            abstract = abstract.to(self.device)
            GA = GA.to(self.device)
            caption = caption.to(self.device) if self.is_merge_caption else None

            # Forward
            with torch.amp.autocast(self.device.type):
                output: models.Abs2FigRetrieverOutputForInterGARecommendation = self.model(
                    abstract,
                    GA,
                    caption,
                )
                loss = output.inter_loss
                abstract_embed = output.abstract_embed

                normalized_abstract_embed = abstract_embed / abstract_embed.norm(dim=-1, keepdim=True)
                normalized_GA_embed = candidate_GA_embed / candidate_GA_embed.norm(dim=-1, keepdim=True)

                if type(self.model) == models.BLIP2AsAbs2FigRetrieverForInterGARecommendation:
                    sim_GA2abs = torch.matmul(normalized_abstract_embed, normalized_GA_embed.transpose(-2, -1))
                    sim_GA2abs, _ = sim_GA2abs.max(dim=-1)
                    sim_abs2GA = sim_GA2abs.T
                else:
                    sim_abs2GA = torch.matmul(normalized_abstract_embed, normalized_GA_embed.T)

                probs, preds = sim_abs2GA.sort(dim=-1, descending=True)
                probs = probs.squeeze(1)[:, :10]
                preds = preds.squeeze(1)[:, :10]

            # Save loss and result
            epoch_loss += loss.item()
            for j, id in enumerate(paper_id):
                for k, prob in zip(preds[j], probs[j]):
                    k = k.item()
                    prob = prob.item()
                    epoch_result.append({
                        'paper_id': id,
                        'research_fields': research_fields[j],
                        'retrieved_paper_id': candidate_paper_id[k],
                        'retrieved_research_fields': candidate_research_fields[k],
                        'prob': prob,
                    })

        # Record average loss
        epoch_loss = epoch_loss / len(dataloader)

        # Record score
        epoch_result = pd.DataFrame(epoch_result)
        mean_field_precision, mean_abs2abs_SBERT, std_abs2abs_SBERT, mean_GA2GA_CLIPScore, std_GA2GA_CLIPScore, mean_GA2GA_DreamSim, std_GA2GA_DreamSim = metrics.evaluate_interGA_recommendation_metrics(
            epoch_result,
            SBERT_embeddings=self.SBERT_embeddings,
            CLIP_embeddings=self.CLIP_embeddings,
            DreamSim_embeddings=self.DreamSim_embeddings,
        )
        epoch_score = {
            'Field-P@1': None,
            'Field-P@5': mean_field_precision['5'],
            'Field-P@10': mean_field_precision['10'],
            'Abs2Abs_SBERT@1': None,
            'Abs2Abs_SBERT@5': (mean_abs2abs_SBERT['5'], std_abs2abs_SBERT['5']),
            'Abs2Abs_SBERT@10': (mean_abs2abs_SBERT['10'], std_abs2abs_SBERT['10']),
            'GA2GA_CLIP-S@1': None,
            'GA2GA_CLIP-S@5': (mean_GA2GA_CLIPScore['5'], std_GA2GA_CLIPScore['5']),
            'GA2GA_CLIP-S@10': (mean_GA2GA_CLIPScore['10'], std_GA2GA_CLIPScore['10']),
            'GA2GA_DreamSim@1': None,
            'GA2GA_DreamSim@5': (mean_GA2GA_DreamSim['5'], std_GA2GA_DreamSim['5']),
            'GA2GA_DreamSim@10': (mean_GA2GA_DreamSim['10'], std_GA2GA_DreamSim['10']),
        }

        return epoch_loss, epoch_score


def main(args: Args) -> None:
    is_wandb = args.is_wandb
    experiment_name = args.experiment_name
    seed = args.seed
    dataset_json_dir = args.dataset_json_dir
    dataset_figure_dir = args.dataset_figure_dir
    save_cache_dir = args.save_cache_dir
    model_type = args.model_type
    model_name = args.model_name or None
    model_config_path = args.model_config_path or None
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
    full_split = datasets.load_and_prepare_split(
        split_name='dataset',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
    )
    train = datasets.load_and_prepare_split(
        split_name='train',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
    )
    valid = datasets.load_and_prepare_split(
        split_name='valid',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
    )
    test = datasets.load_and_prepare_split(
        split_name='test',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
    )
    print(f'   train: {len(train)}, valid: {len(valid)}, test: {len(test)}')

    # Load model
    print(f'\n📦 Loading model({model_type})...')
    model = model_loader.load_abs2fig_retriever_for_inter_GA_recommendation(
        model_type=model_type,
        model_name=model_name,
        model_config_path=model_config_path,
    )

    # Create Dataset
    train_dataset = datasets.Abs2FigRetrieverDatasetForInterGARecommendation(
        paper_id=train['paper_id'].tolist(),
        research_fields=train['research_fields'].tolist(),
        abstract=train['abstract'].tolist(),
        GA_path=train['GA_path'].tolist(),
        GA_caption=train['GA_caption'].tolist(),
        tokenizer=model.tokenize,
        transform=model.preprocess_image,
    )
    valid_dataset = datasets.Abs2FigRetrieverDatasetForInterGARecommendation(
        paper_id=valid['paper_id'].tolist(),
        research_fields=valid['research_fields'].tolist(),
        abstract=valid['abstract'].tolist(),
        GA_path=valid['GA_path'].tolist(),
        GA_caption=valid['GA_caption'].tolist(),
        tokenizer=model.tokenize,
        transform=model.preprocess_image,
    )
    test_dataset = datasets.Abs2FigRetrieverDatasetForInterGARecommendation(
        paper_id=test['paper_id'].tolist(),
        research_fields=test['research_fields'].tolist(),
        abstract=test['abstract'].tolist(),
        GA_path=test['GA_path'].tolist(),
        GA_caption=test['GA_caption'].tolist(),
        tokenizer=model.tokenize,
        transform=model.preprocess_image,
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
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    candidate_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load or Compute SBERT embeddings for Abstracts
    # NOTE: Use cached embeddings for fast Abs2Abs SBERT@k metric evaluation
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'\n🚀 Using device: {device}')

    cache_path = f'{save_cache_dir}sbert_embeddings.pt'
    if not os.path.exists(cache_path):
        print(f'\n💾 Encode and saving SBERT text embeddings to: \'{cache_path}\'...')
        SBERT_embeddings = metrics.save_SBERT_embeddings(
            paper_ids=full_split['paper_id'].tolist(),
            abstracts=full_split['abstract'].tolist(),
            cache_path=cache_path,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            device=device,
            batch_size=batch_size,
        )
    else:
        print(f'\n📦 Loading SBERT text embeddings from: \'{cache_path}\'...')
        SBERT_embeddings = torch.load(cache_path)

    # Load or Compute CLIP embeddings for GA Images
    # NOTE: Use cached embeddings for fast GA2GA CLIPScore@k metric evaluation
    cache_path = f'{save_cache_dir}clip_embeddings.pt'
    if not os.path.exists(cache_path):
        print(f'\n💾 Encode and saving CLIP text embeddings to: \'{cache_path}\'...')
        CLIP_embeddings = metrics.save_CLIP_embeddings(
            paper_ids=full_split['paper_id'].tolist(),
            GA_paths=full_split['GA_path'].tolist(),
            cache_path=cache_path,
            model_name='ViT-L/14',
            device=device,
            batch_size=batch_size,
        )
    else:
        print(f'\n📦 Loading CLIP image embeddings from: \'{cache_path}\'...')
        CLIP_embeddings = torch.load(cache_path)

    # Load or Compute DreamSim embeddings for GA Images
    # NOTE: Use cached embeddings for fast GA2GA DreamSim@k metric evaluation
    cache_path = f'{save_cache_dir}dreamsim_embeddings.pt'
    if not os.path.exists(cache_path):
        print(f'\n💾 Encode and saving DreamSim text embeddings to: \'{cache_path}\'...')
        DreamSim_embeddings = metrics.save_DreamSim_embeddings(
            paper_ids=full_split['paper_id'].tolist(),
            GA_paths=full_split['GA_path'].tolist(),
            cache_path=cache_path,
            device=device,
            batch_size=batch_size,
        )
    else:
        print(f'\n📦 Loading DreamSim image embeddings from: \'{cache_path}\'...')
        DreamSim_embeddings = torch.load(cache_path)

    # Run experiments
    experiment = Experiment(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        candidate_dataloader=candidate_dataloader,
        SBERT_embeddings=SBERT_embeddings,
        CLIP_embeddings=CLIP_embeddings,
        DreamSim_embeddings=DreamSim_embeddings,
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
