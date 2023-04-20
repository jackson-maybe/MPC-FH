# MPC-FH: Privately Estimating Frequency Histogram for Advertising Measurement
This repository includes our realization of MPC-FH, which is a secure protocol for efficiently estimating the frequency histogram in the advertising measurement, i.e. the fraction of persons (or users) appearing a given number of times across all publishers. Though MPC-LiquidLegions is a way to solve the problem, however, the protocol demands that the aggregator and workers used are assumed to be honest but curious, and it has an expensive computational cost. Our MPC-FH is based on a novel sketch method MPS++ which consistently and uniformly samples active users distributed over multiple publishers. The salient strength of our MPS++ is that it can be efficiently implemented on MPC platforms including SPDZ, which can guarantee security when computational parties are malicious. Our experimental results show that our MPC-FH accelerates the computational speed of the state-of-the-art protocol by 20–87 times.
## Datasets
To simulate the real scenario, we first generate varieties of publishers' ID datasets, which are randomly sampled from the universe set without duplicates. As for the users' frequency datasets, we set the homogeneous and heterogeneous case. For the homogeneous case the frequencies across different users are sampled from the same zero-shifted Poisson distribution, while for the heterogeneous case each of them is sampled from a customized zero-shifted negative binomial distribution. 
## Metric
To evaluate the accuracy, we use the metric Shuffle Distance (SD) to measure the difference between the estimated frequency histogram and the ground-truth frequency histogram. We use the metric because it represents the fraction of data whose frequency is wrong.
In addition, we measure the running time and communication overload of our protocol MPC-FH and the state-of-the-art protocol MPC-LL.
## Methods
| Method             | Data Structure        | Sensitivity | MPC Security Model |
| -----------        | -----------           | ----------- | --------------     |
| MPC-FM(MPS)        | 4M integer arrays     |M            | Malicious          |
| MPC-FM(MPS++)      | 4M integer arrays     |1            | Malicious          |
| MPC-LiquidLegions  | 3 integer arrays      |2            | Honest but curious |
## Accuracy Experiments
Run "pip install -r requirements.txt' to download required packages.
Then, if you want to test the accuracy of MPC-FM (using MPS sketch), please run '''python python MPS.py''''
if you want to test the accuracy of MPC-FM (using MPS++ sketch), please run 'python python MPS++.py'
if you want to test the accuracy of MPC-LiquidLegions, please run python 'python LiquidLegions.py'
