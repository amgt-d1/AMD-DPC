## Introduction
* This repository provides implementation of AMD-DPC.
* This is a fast approximation algorithm for [density-peaks clustering](https://science.sciencemag.org/content/344/6191/1492.full) (proposed in Science) on fully dynamic data in any metric space.
* As for the details about AMD-DPC, please read our IEEE BigData 2022 paper, [Scalable and Accurate Density-Peaks Clustering on Fully Dynamic Data](https://).

## Datasets
* [Gas](https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+under+dynamic+gas+mixtures)
* [Household](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
* [PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
* [Mirai](https://archive.ics.uci.edu/ml/datasets/Kitsune+Network+Attack+Dataset)
* Our code allows a .csv file (each row is a d-dimensional vector where each dimension is separeted by comma) as input.

## Parameter Setting
* See `amd-dpc/parameter` directory.
* If you want to test a distance function other than L1 and L2, you can implement the function in `compute_distance()` in `data.hpp`.

## How to Run
* Compile: `g++ -O3 -o amd-dpc.out main.cpp --std=c++14 -fopenmp -Wall`
   * .out file name can be arbitrary.
* Run: `./amd-dpc.out`
* We used Ubuntu 18.04 LTS.

## Result
* Creat `result` directory.
* For each dataset, the experimental result is provided in the corresponding directory.
   * Dataset IDs of Gas, Household, PAMAP2, and Mirai are 3, 1, 2, and 4, respectively.
   * Create `result/x-y`, where x is dataset ID and y is dataset name.
* We calculated accuracy (e.g., NMI) by using scikit-learn library.
   * As our algorithm maintains (approximate) local density and dependent point for each object, it is straightforward to implement codes for obtaining the clustering result of any time.

## Citation
If you use our implementation, please cite the following paper.
``` 
@inproceedings{amagata2022scalable,  
    title={Fast Density-Peaks Clustering: Multicore-based Parallelization Approach},  
    author={Amagata, Daichi},  
    booktitle={IEEE BigData},  
    pages={xx--xx},  
    year={2022}  
}
```

## License
Copyright (c) 2022 Daichi Amagata  
This software is released under the [MIT license](https://github.com/amgt-d1/AMD-DPC/blob/main/license.txt).
