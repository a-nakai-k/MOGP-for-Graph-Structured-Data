# MOGP-for-Graph-Structured-Data

A Python implementation of multi-output Gaussian process regression for graph-structured data.

A. Nakai-Kasai and T. Wadayama, ''[Multi-Output Gaussian Processes for Graph-Structured Data](https://arxiv.org/abs/2505.16755),''  arXiv preprint, arXiv:2505.16755 [cs.LG], May 2025. 


## Requirement
Please refer to conda_requirements.txt.

## Usage
Run:
```
pyhom3 baselines_synthetic1.py --model=standard --kernel=rbf --k=12
```
```
pyhom3 baselines_synthetic2.py --model=standard --kernel=rbf
```

## License
This project is licensed under the MIT License, see the LICENSE file for details.

## Author
[Ayano NAKAI-KASAI](https://sites.google.com/view/ayano-nakai/home/english)

Graduate School of Engineering, Nagoya Institute of Technology

nakai.ayano@nitech.ac.jp

## Acknowledgment
Some parts of the functions are based on the implementation by Yin-Cong Zhi (Zhi et al., 2023).

## References
Y.-C. Zhi, Y. C. Ng, and X. Dong, ”Gaussian processes on graphs via
spectral kernel learning,” IEEE Trans. Signal Inf. Process. Netw., vol. 9,
pp. 304–314, Apr. 2023. [GitHub](https://github.com/yincong-zhi/Polynomial_Kernel_Learning)
