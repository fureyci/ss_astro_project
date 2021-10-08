# ss_astro_project

# Ensemble Techniques for Space Weather Forecasting
## Ciaran Furey, Trinity College Dublin
### Final Year Astrophysics Capstone Research Project, 2021/22

This repository will store any code/files that I use throughout the course of my project.

My goal is to build an ensemble model in python in order to provide more accurate forecasts of solar flares. This will consist of many individual models, namely, those referenced in [this paper](https://iopscience.iop.org/article/10.3847/1538-4365/ab2e12/pdf) (Leka et al., 2019). 

As of now, I have been getting used to the type of data I will be working with. This consists of the forecasts of M- and C-class solar flares provided by different models between 1/1/2016 and 31/12/2017, and was accessed [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HYP74O) (Leka and Park, 2019). 

##### 29/09/2021 Update
Upon further reading of Leka et al., 2019, I found out that when calculating the fraction of positives for the reliability curves, Laplace's rule of succession is used (Wheatland, 2005). This has been implemented in `metric_utils.py` by updating the function `plot_reliability_curve()`. Now that any model can be compared to the benchmark, the next goal is to update `ensemble.py`. 

##### 08/10/2021 Update
Updated `ensemble.py` by including a constrained linear combination weighting scheme. Still need to debug the unconstrained linear combination weighting scheme. Also added a file `ens_test.py` to test the functionality of `ensemble.py`. Still need to debug the unconstrained linear combination weighting scheme, as well as some other things.

###### References 
* Leka, K.D., Park, S.H., Kusano, K., Andries, J., Barnes, G., Bingham, S., Bloomfield, D.S., McCloskey, A.E., Delouille, V., Falconer, D. and Gallagher, P.T., 2019. A comparison of flare forecasting methods. II. Benchmarks, metrics, and performance results for operational solar flare forecasting systems. The Astrophysical Journal Supplement Series, 243(2), p.36.

* Leka, K. D.; Park, Sung-Hong, 2019, "A Comparison of Flare Forecasting Methods II: Data and Supporting Code", [https://doi.org/10.7910/DVN/HYP74O], Harvard Dataverse, V1, UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]

* Wheatland, M.S., 2005. A statistical solar flare forecast method. Space Weather, 3(7).

* `metric_utils.py` obtained from https://github.com/hayesla/flare_forecast_proj/blob/main/forecast_tests/metric_utils.py

