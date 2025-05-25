import os
import pandas as pd
import random
import torch
import benchmark.utils as utils
import benchmark.datasets as datasets
import benchmark.metrics as metrics
from tap import Tap
from tqdm import tqdm
import wandb


class Args(Tap):
    experiment_name: str = 'SciGA_interGA_RandomSampling'
    seed: int = 42
    device: int = 0
    dataset_json_dir: str = './benchmark/SciGA_for_experiments/'
    dataset_figure_dir: str = './SciGA/'
    save_cache_dir: str = './benchmark/output/caches/'

    batch_size: int = 16

    is_wandb: bool = False

    def configure(self):
        self.add_argument('--is_wandb', action='store_true', help='Use Weights and Biases for logging', required=False)


class Experiment():
    def __init__(
        self,
        candidate_df: pd.DataFrame,
        SBERT_embeddings: dict[str, torch.Tensor],
        CLIP_embeddings: dict[str, torch.Tensor],
        DreamSim_embeddings: dict[str, torch.Tensor],
        args: Args
    ) -> None:
        self.experiment_name = args.experiment_name
        self.candidate_df = candidate_df
        self.SBERT_embeddings = SBERT_embeddings
        self.CLIP_embeddings = CLIP_embeddings
        self.DreamSim_embeddings = DreamSim_embeddings
        self.is_wandb = args.is_wandb

    @torch.inference_mode()
    def test(self, test_df: pd.DataFrame, desc: str) -> dict[str, float | tuple[float, float]]:
        paper_ids = test_df['paper_id'].tolist()
        research_fields = test_df['research_fields'].tolist()
        candidate_paper_ids = self.candidate_df['paper_id'].tolist()
        candidate_fields = self.candidate_df['research_fields'].tolist()

        epoch_result = []

        for j, paper_id in enumerate(tqdm(paper_ids, desc='   ' + desc, ncols=80)):
            # Perform retrieval
            random_indices = random.sample(range(len(self.candidate_df)), 10)

            # Save result
            for k in random_indices:
                epoch_result.append({
                    'paper_id': paper_id,
                    'research_fields': research_fields[j],
                    'retrieved_paper_id': candidate_paper_ids[k],
                    'retrieved_research_fields': candidate_fields[k],
                    'prob': 0.0,
                })

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

        return epoch_score


def main(args: Args) -> None:
    is_wandb = args.is_wandb
    experiment_name = args.experiment_name
    seed = args.seed
    dataset_json_dir = args.dataset_json_dir
    dataset_figure_dir = args.dataset_figure_dir
    save_cache_dir = args.save_cache_dir
    batch_size = args.batch_size

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
    test = datasets.load_and_prepare_split(
        split_name='test',
        dataset_json_dir=dataset_json_dir,
        dataset_figure_dir=dataset_figure_dir,
    )
    print(f'   train: {len(train)}, test: {len(test)}')

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

    # Load or Compute CLIP embeddings for GAs
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

    # Load or Compute DreamSim embeddings for GAs
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
        candidate_df=train,
        SBERT_embeddings=SBERT_embeddings,
        CLIP_embeddings=CLIP_embeddings,
        DreamSim_embeddings=DreamSim_embeddings,
        args=args
    )

    print('\n🧪 Running Test Evaluation...')
    score = experiment.test(test_df=test, desc='test')
    utils.print_epoch_scores(
        phases=['test'],
        losses=[None],
        scores=[score]
    )


if __name__ == '__main__':
    args = Args().parse_args()
    main(args)
