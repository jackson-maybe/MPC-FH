# MPC-FH: Privately Estimating Frequency Histogram for Advertising Measurement
This repository includes our realization of MPC-FH, which is a secure protocol for efficiently estimating the frequency histogram in the advertising measurement, i.e. the fraction of persons (or users) appearing a given number of times across all publishers. Though MPC-LiquidLegions is a way to solve the problem, however, the protocol demands that the aggregator and workers used are assumed to be honest but curious, and it has an expensive computational cost. Our MPC-FH is based on a novel sketch method MPS++ which consistently and uniformly samples active users distributed over multiple publishers. The salient strength of our MPS++ is that it can be efficiently implemented on MPC platforms including SPDZ, which can guarantee security when computational parties are malicious. Our experimental results show that our MPC-FH accelerates the computational speed of the state-of-the-art protocol by 20–87 times.
## Datasets
To simulate the real scenario, we first generate varieties of publishers' ID datasets $U_i$, which are randomly sampled from the universe set $\mathcal{U}$ without duplicates and the sampling proportion is $P \in (0,1]$. The size of the universe set $|\mathcal{U}|$ varies across $\{10^3, 10^5, 10^7, 10^9 \}$, the number of publishers varies across $\{5,10,15,20,25\}$ and the sampling proportion $P$ varies across $\{0.01,0.05,0.1,0.15,0.20\}$.

As for the users' frequencies $f_u^{(i)}$, we set the homogeneous and heterogeneous case. For the homogeneous case the frequencies $f_u^{(i)}$ across different $i$ and $u$ are sampled from the same zero-shifted Poisson distribution, i.e., $f_u^{(i)} \sim Poisson(\lambda)+1$, while for the heterogeneous case each of them is sampled from a customized zero-shifted negative binomial distribution, i.e., $f_u^{(i)} \sim \binom{f_u^{(i)}+r-1}{f_u^{(i)}}(\frac{\alpha}{\alpha+1})^r (\frac{1}{\alpha+1})^{f_u^{(i)}}+1$. The heterogeneous case means that each $f_u^{(i)}$ comes from a customized gamma distribution, i.e., $f_u^{(i)} \sim Poisson(\lambda_i)+1$ and $\lambda_i \sim \frac{\alpha^{r}\lambda_i^{r-1}e^{-\alpha\lambda_i}}{\Gamma(r)}$, which represents the heterogeneity of publishers' datasets. Without loss of generality, we set $\lambda = 1$ for the homogeneous case and $r=1, \alpha=1$ for the heterogeneous case.
## Metrics
To evaluate the accuracy, we use the metric \emph{Shuffle Distance} (SD, used in~\cite{WFA2020}) to measure the difference between the estimated frequency histogram 
$\vec{\hat{\pi}}$ and the ground-truth frequency histogram $\vec{\pi}$. SD is formally defined as SD($\vec{\hat{\pi}}$, $\vec{\pi}$) =$\frac{1}{2}$ $\sum_{l=1}^{15+}$|$\hat{\pi}_l$-$\pi_l$|, where $\vec{\hat{\pi}}= \{\hat{\pi_l}, l=1,2,\ldots, 15+\}, \vec{\pi}= \{\pi_l, l=1,2,\ldots, 15+\}$.We use the metric because it represents the fraction of data whose frequency is wrong. In addition, we measure the running time and communication overload of our protocol MPC-FH and the state-of-the-art protocol MPC-LL in~\cite{GhaziKKMPSWW22}. All experimental results are averaged over 10 independent runs.
