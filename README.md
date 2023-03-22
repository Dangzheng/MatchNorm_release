# MatchNorm

This repository contains python scripts of paper [**Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World**, ECCV2022](https://arxiv.org/pdf/2203.15309.pdf).

## Prerequisites

1. [BOP Toolkit](https://github.com/thodan/bop_toolkit) before running the code please setup the path for loading the BOP Benchmark data.

2. [TUD-L, LINEMOD and Occluded-LINEMOD](https://bop.felk.cvut.cz/datasets/)

3. conda env create -f environment.yml

## Command Line

### Training
    python scratch.py --mode='ori' --bop_dataset='tudl'

### Testing

    [mAP Metric]

    python scratch.py --mode='ckpt' --bop_dataset='tudl' --exp_name='bpnet_tudl'

    python scratch.py --mode='ckpt' --bop_dataset='lm' --exp_name='bpnet_lm'

    python scratch.py --mode='ckpt' --bop_dataset='lmo' --exp_name='bpnet_lm'

    [BOP Metric]

    python eval_bop19.py --result_filenames='zheng-bpnet_tudl-test.csv'
    
    python eval_bop19.py --result_filenames='zheng-bpnet_lm-test.csv'

    python eval_bop19.py --result_filenames='zheng-bpnet_lmo-test.csv'

    
## Pretrained Model

1. pretrained models are saved at `./ckpt`

2. bop benchmark result files are saved at `./bop_res`

## Citation

    @inproceedings{Zheng2022,
        title={Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World},
        author={Dang, Zheng and Wang, Lizhou and Guo, Yu and Salzmann, Mathieu},
        booktitle = {European Conference on Computer Vision(ECCV) 2022},
        month = {October},
        year={2022}
    }

## License
MIT License