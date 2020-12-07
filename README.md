# PABI
This is the code repository for the ArXiv paper [Foreshadowing the Benefits of Incidental Supervision](https://arxiv.org/pdf/2006.05500.pdf).
If you use this code for your work, please cite
```
@article{he2020foreshadowing,
  title={Foreshadowing the Benefits of Incidental Supervision},
  author={He, Hangfeng and Zhang, Mingyuan and Ning, Qiang and Roth, Dan},
  journal={arXiv preprint arXiv:2006.05500},
  year={2020}
}
```


## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
pip install -r requirements.txt

## Code Organization

The code is organized as follows:
- bpp.py (CWBPP algorithm for learning with various inductive signals)
- run_ner.py (BERT for NER)
- run_squad.py (BERT for QA)


## Reproducing experiments
To reproduce the experiments for learning with various inductive signals:
```
sh run_experiments.sh
```

To reproduce the experiments for cross-domain signals:
```
sh run_xdomain_ner_experiments.sh
sh run_xdomain_qa_experiments.sh
```

