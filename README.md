
# SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers

[![arXiv](https://img.shields.io/badge/arXiv-2507.02212-b31b1b.svg)](https://arxiv.org/abs/2507.02212)
[![Huggingface Datasets](https://img.shields.io/badge/🤗_Huggingface-Dataset-yellow.svg)](https://huggingface.co/datasets/IyatomiLab/SciGA)
[![Project Page](https://img.shields.io/badge/Project_Page-SciGA-green.svg)](https://iyatomilab.github.io/SciGA/)


Takuro Kawada,
[Shunsuke Kitada](https://shunk031.me/),
Sota Nemoto,
[Hitoshi Iyatomi](https://iyatomi-lab.info/english-top)



Graphical Abstracts (GAs) play a crucial role in visually conveying the key findings of scientific papers. While recent research has increasingly incorporated visual materials such as Figure 1 as de facto GAs, their potential to enhance scientific communication remains largely unexplored. Moreover, designing effective GAs requires advanced visualization skills, creating a barrier to their widespread adoption. To tackle these challenges, we introduce SciGA-145k, a large-scale dataset comprising approximately 145,000 scientific papers and 1.14 million figures, explicitly designed for supporting GA selection and recommendation as well as facilitating research in automated GA generation. As a preliminary step toward GA design support, we define two tasks: 1) Intra-GA recommendation, which identifies figures within a given paper that are well-suited to serve as GAs, and 2) Inter-GA recommendation, which retrieves GAs from other papers to inspire the creation of new GAs. We provide reasonable baseline models for these tasks. Furthermore, we propose Confidence Adjusted top-1 ground truth Ratio (CAR), a novel recommendation metric that offers a fine-grained analysis of model behavior. CAR addresses limitations in traditional ranking-based metrics by considering cases where multiple figures within a paper, beyond the explicitly labeled GA, may also serve as GAs. By unifying these tasks and metrics, our SciGA-145k establishes a foundation for advancing visual scientific communication while contributing to the development of AI for Science.



## 📰 News

🚀 [2024/07/03] Dataset, code, and models released<br>
🚀 [2024/07/03] Paper available on arXiv<br>



## 🐐 SciGA Dataset
**SciGA**  is a large-scale dataset designed to support research in GA generation, recommendation, and visual understanding of scientific papers.

### Contents
- 145k scientific papers (from arXiv)
- 1.1M figures (`.png`, `.mp4`)
- 150+ GAs — author-provided visual summaries, annotated with type
- 30,000+ teasers — first-page figures that serve as de facto GAs
- Structured metadata (sections, figures, DOIs, subjects, etc.)
- Annotated GA types: `Original`, `Reused`, `Modified`

### Usage
``` Python
import datasets as ds

dataset = ds.load_dataset('Iyatomilab/SciGA')
```



## 🔍 GA Recommendation Benchmarks
SciGA supports two GA recommendation tasks:
- Intra-GA Recommendation: Select figures within a paper that are suitable as GAs.
- Inter-GA Recommendation: Retrieve GAs from other papers as design inspiration.

### Installation
``` bash
$ git clone git@github.com:IyatomiLab/SciGA.git
$ cd SciGA

# Prepare submodules
$ git submodule update --init --recursive
$ git -C benchmark/submodules/x2vlm apply ../patches/x2vlm.patch
$ cat SciGA-for-experiments/SciGA-part-* > SciGA-for-experiments/SciGA-for-experiments.tar.gz
$ tar -xzvf SciGA-for-experiments/SciGA-for-experiments.tar.gz
```

### Demo
Try our pretrained models using the provided notebooks:
👉 [demo.ipynb](https://github.com/IyatomiLab/SciGA/blob/main/demo.ipynb)


### Train
``` bash
# Intra-GA Recommendation
# $ bash benchmark/scripts/run_intra_with_abs2cap.sh
# $ bash benchmark/scripts/run_intra_with_ga_bc.sh
$ bash benchmark/scripts/run_intra_with_abs2fig.sh

# Inter-GA Recommendation
# $ bash benchmark/scripts/run_inter_with_random.sh
# $ bash benchmark/scripts/run_inter_with_abs2cap.sh
$ bash benchmark/scripts/run_inter_with_abs2fig.sh
```
You can switch model backbones (e.g., CLIP, Long-CLIP, BLIP-2) by editing the bash files.



## 📐 CAR
**CAR@*k*** (Confidence-Adjusted Top-1 Ground Truth Ratio) is a metric designed for top-*k* recommendation tasks where multiple candidates can be plausible answers. <br>
It combines prediction confidence with ranking performance, providing a more nuanced evaluation than traditional metrics like Recall@*k*.

CAR@*k* is task-agnostic and can be used in any ranking-based evaluation pipeline where soft ground truth matching and model confidence matter.

### Usage
``` bash
# Install CAR module (local version)
$ git clone git@github.com:IyatomiLab/SciGA.git
$ cd SciGA
$ pip install -e ./car
```

```  python
from car import confidence_adjusted_top1_gt_ratio

# Prediction relevance scores for 10 candidate items
# NOTE: Raw scores (e.g., similarity, logits) are recommended to better reflect confidence
#       Already-softmaxed probabilities are also acceptable, though less precise.
scores = [0.12, 0.35, 0.05, 0.18, 0.03, 0.44, 0.09, 0.07, 0.25, 0.29]

# Index of the ground truth item in the original score list
# NOTE: This assumes the correct answer is the 6th candidate (score = 0.44)
gt_index = 5

# Top-k scope to evaluate CAR
k = 5

# Compute CAR@5
car = confidence_adjusted_top1_gt_ratio(scores, k, gt_index)
print(f"CAR@{k} = {car:.3f}")
```

> [!NOTE]  
> We plan to release car on PyPI. Stay tuned!

## ✨ Citation
If you find our work helpful, please cite:
``` bibtex
@article{kawada2025sciga,
        title={SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers},
        author={Takuro Kawada and Shunsuke Kitada and Sota Nemoto and Hitoshi Iyatomi},
        journal={arXiv preprint arXiv:2507.02212},
        year={2025}
}
```