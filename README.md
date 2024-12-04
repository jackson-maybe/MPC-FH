# MPC-FH: Privately Estimating Frequency Histogram for Advertising Measurement
This repository includes our realization of MPC-FH, which is a secure protocol for efficiently estimating the frequency histogram in the advertising measurement, i.e. the fraction of persons (or users) appearing a given number of times across all publishers. Though MPC-LiquidLegions is a way to solve the problem, however, the protocol demands that the aggregator and workers used are assumed to be honest but curious, and it has an expensive computational cost. Our MPC-FH is based on a novel sketch method MPS which consistently and uniformly samples active users distributed over multiple publishers. The salient strength of our MPS is that it can be efficiently implemented on MPC platforms including SPDZ, which can guarantee security when computational parties are malicious. Our experimental results show that our MPC-FH accelerates the computational speed of the state-of-the-art protocol by 20–87 times.
## Datasets
To simulate the real scenario, we first generate varieties of publishers' ID datasets, which are randomly sampled from the universe set without duplicates. As for the publishers' frequency datasets, we set the homogeneous and heterogeneous cases. For the homogeneous case, the frequencies across different users are sampled from the same zero-shifted Poisson distribution, while for the heterogeneous case, each of them is sampled from a customized zero-shifted negative binomial distribution. 
## Metric
To evaluate the accuracy, we use the metric Shuffle Distance (SD) to measure the difference between the estimated frequency histogram and the ground-truth frequency histogram. We use the metric because it represents the fraction of data whose frequency is wrong.
In addition, we measure the running time and communication overload of our protocol MPC-FH compared with the state-of-the-art protocol MPC-LiquidLegions.
## Methods
| Protocol            | Computation Cost      | MPC Security Model |
| :---------:         | :---------:           | :------------:     |
| MPC-FH              |Low                    | Malicious          |
| MPC-LiquidLegions   |High                   | Honest-but-Curious |
## Experiments for Accuracy
1. Run `pip install -r requirements.txt` to download the required packages.

2. If you want to test the accuracy of MPC-FH (using MPS sketch), please run `python MPS.py`.

   If you want to test the accuracy of MPC-LiquidLegions, please run `python LiquidLegions.py`.
## Experiments for computational costs
1. Run `python MPS.py` to generate the MPS sketch of each publisher.
2. Prepare requirements and compile fundamental protocols of SPDZ.

```
cd MP-SPDZ
sudo apt-get install automake build-essential clang cmake git libboost-dev libboost-thread-dev libntl-dev libsodium-dev libssl-dev libtool m4 python3 texinfo yasm libgmp-dev libmpfr-dev libmpc-dev
sh protocol_compile.sh
```

   Please make sure that CMake's version is at least 3.12 .
   
3. Compile our MPC-FH protocol and set the number of computation providers (CPs).

```
./compile.py -F 64 MPC_FH
Scripts/setup-online.sh <nCPs>
```

   Besides, we should also set the number of CPs in the `Scripts/run-common.sh`.
   
4. Run our MPC-FH protocol.

```
Scripts/run-online.sh MPC_FH
```
   
