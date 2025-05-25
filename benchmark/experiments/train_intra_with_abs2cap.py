import pandas as pd
import re
import torch
import benchmark.models as models
import benchmark.models.matcher_loader as model_loader
import benchmark.datasets as datasets
import benchmark.metrics as metrics
import benchmark.utils as utils
from tap import Tap
from tqdm import tqdm
import wandb


class Args(Tap):
    experiment_name: str = 'SciGA_intraGA_Abs2Cap'
    seed: int = 42
    dataset_json_dir: str = './benchmark/SciGA_for_experiments/'

    model_type: str = 'ROUGE'

    is_wandb: bool = False

    def configure(self):
        self.add_argument('--is_wandb', action='store_true', help='Use Weights and Biases for logging', required=False)


class Experiment():
    def __init__(
        self,
        model: models.BaseAbs2CapMatcher,
        args: Args
    ) -> None:
        self.experiment_name = args.experiment_name
        self.model = model
        self.is_wandb = args.is_wandb

    @torch.inference_mode()
    def test(self, test_df: pd.DataFrame, desc: str) -> dict[str, float | tuple[float, float]]:
        # Preprocess data
        test_df['captions'] = test_df.apply(
            lambda x: [x['GA_caption']] + x['figure_captions'],
            axis=1
        )
        test_df['figure_ids'] = test_df.apply(
            lambda x:
            ['GA'] + [
                re.match(r'.*_(F\d+)(?:\.\d+|\(\d+\))?.*', path).group(1) for path in x['figure_paths']
            ],
            axis=1
        )

        paper_ids = test_df['paper_id'].tolist()
        figure_ids = test_df['figure_ids'].tolist()
        GT_figure_ids = test_df['GT_figure_ids'].tolist()

        epoch_result = []

        # Perform retrieval
        tqdm.pandas(desc='   ' + desc, ncols=80)
        output = test_df.progress_apply(lambda x: self.model.match(x['abstract'], x['captions']), axis=1)

        output = output.apply(lambda x: x.sim_abs2cap)
        output = output.tolist()

        # Save result
        for j, paper_id in enumerate(paper_ids):
            sim_abs2cap = torch.tensor(output[j])
            probs, preds = sim_abs2cap.sort(dim=-1, descending=True)
            for k, prob in zip(preds, probs):
                prob = prob.item()
                k = k.item()
                figure_id = figure_ids[j][k]
                epoch_result.append({
                    'paper_id': paper_id,
                    'figure_id': figure_id,
                    'prob': prob,
                    'GT_figure_ids': GT_figure_ids[j],
                })

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

        return epoch_score


def main(args: Args) -> None:
    is_wandb = args.is_wandb
    experiment_name = args.experiment_name
    seed = args.seed
    dataset_json_dir = args.dataset_json_dir
    model_type = args.model_type

    print(f'\n🔍 Experiment Name: {experiment_name}')

    # Set up Weights and Biases
    if is_wandb:
        wandb.login()
        wandb.init(project='SciGA', name=f'{experiment_name}', config=args.__dict__)

    # Set seed
    utils.set_seed(seed)

    # Load data
    print(f'\n📦 Loading dataset from: \'{dataset_json_dir}\'...')
    test = datasets.load_and_prepare_split(
        split_name='test',
        dataset_json_dir=dataset_json_dir,
    )
    print(f'   test: {len(test)}')

    # Load model
    print(f'\n📦 Loading model({model_type})...')
    model = model_loader.load_abs2cap_matcher(
        model_type=model_type,
    )

    # Run experiments
    experiment = Experiment(
        model=model,
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
