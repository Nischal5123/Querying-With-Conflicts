# Querying with Conflicts of Interest

Conflicts of interest often arise between data sources and their
users regarding how the users’ information needs should be inter-
preted by the data source. For example, an online product search
might be biased towards presenting certain products higher than
in its list of results to improve its revenue, which may not follow
the user’s desired ranking expressed in their query. The research
community has proposed schemes for data systems to implement
to ensure unbiased results. However, data systems and services usu-
ally have little or no incentive to implement these measures, e.g.,
these biases often increase their profits. In this paper, we propose a
novel formal framework for querying in settings where the data
source has incentives to return biased answers intentionally due
to the conflict of interest between the user and the data source.
We propose efficient algorithms to detect whether it is possible
for users to extract relevant information from biased data sources.
We propose methods to detect biased information in the results
of a query efficiently. We also propose algorithms to reformulate
input queries to increase the amount of relevant information in the
returned results over biased data sources. Using experiments on
real-world datasets, we show that our algorithms are efficient and
return relevant information over large data.

## 🎯 What This Does

This framework implements three algorithms from your research paper:

1. **Algorithm 1**: Detecting Trustworthy Answers 
2. **Algorithm 2-3**: Detecting Influential Queries
3. **Algorithm 4**: Maximally Informative Query (q★) 

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python detect-trustworthy-answers.py
```

### Datasets
Datasets are in \data\real directory. You can also use datasets by placing them in the same directory and updating the file paths in the code.
Due to size Amazon dataset is not included in the repository. You can download it from [Amazon Product Data](https://www.kaggle.com/datasets/aaronfriasr/amazon-products-dataset/data).
