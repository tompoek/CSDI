# CSDI
This is a fork of the github repository for the NeurIPS 2021 paper "[CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)".

Additional contribution is to wrap this algorithm for car following trajectory dataset.

## Requirement

Please install the packages in requirements.txt

## Preparation

Download car following trajectory dataset as directed by [this paper](https://doi.org/10.1016/j.trc.2021.103490). 

## Experiments 

### training and imputation for the trajectory dataset
```shell
python exe_trajectory.py --testmissingratio [missing ratio] --nsample [number of samples]
```

### Visualize results

'visualize_examples_trajectory.ipynb' is a notebook for visualizing results.

### Example results:

Position, speed and acceleration with no missing values:

![image](https://github.com/tompoek/CSDI/assets/41849931/28101255-177c-4b44-89a8-ef47103980e4)

with missing values, our trained model will impute the values with its mean and variance visualized: (performance yet under tuning)

![image](https://github.com/tompoek/CSDI/assets/41849931/b4c68219-15a7-4c0b-af24-820b6f6848a0)


## Citation

The CSDI paper:

```
@inproceedings{tashiro2021csdi,
  title={CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation},
  author={Tashiro, Yusuke and Song, Jiaming and Song, Yang and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
