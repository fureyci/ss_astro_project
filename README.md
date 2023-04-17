# ss_astro_project

# Ensemble Techniques for Space Weather Forecasting
## Ciaran Furey, Trinity College Dublin, Dublin Institude for Advanced Studies
### Final Year Astrophysics Capstone Research Project, 2021/22

This repository will store any code/files that I use throughout the course of my project.

My goal is to build an ensemble model in python in order to provide more accurate forecasts of solar flares. This will consist of many individual models, namely, those referenced in [this paper](https://iopscience.iop.org/article/10.3847/1538-4365/ab2e12/pdf) (Leka et al., 2019). 

As of now (September 2021), I have been getting used to the type of data I will be working with. This consists of the forecasts of M- and C-class solar flares provided by different models between 1 January 2016 to 31 December 2017, and were accessed [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HYP74O) (Leka and Park, 2019). 

### Repository Structure
This repository contains the python scripts developed over the course of this project. However, most are designed not be run, and are simply used in other files in the directory.

* The main model has been implemented in `ensemble.py`.

* To see how the ensemble model works, run `ens_test.py`. This script was used to generate all plots for my thesis, a sample of which can be seen below. 

* The script `realtime_test.py` can be run to see today’s ensemble solar flare forecast, however, this relies heavily on the number of models available on the Flare Scoreboard. This script was also used to create the rolling RPSS plots for my thesis.

* The script `opening_realtime.py` contains the functions that are used to fetch and clean real time data.

* The script `opening_data.py` contains the functions that are used to open the archival forecasts and events as used in the benchmark by Leka et al. 2019.

* The script `metric_utils.py` was developed by `@hayesla` and contains the functions that calculate the different verification metrics.

* The repository `CSV` contains the forecasts and event lists over the period 1 January 2016 to 31 December 2017.


##### 29/09/2021 Update
Upon further reading of Leka et al., 2019, I found out that when calculating the fraction of positives for the reliability curves, Laplace's rule of succession is used (Wheatland, 2005). This has been implemented in `metric_utils.py` by updating the function `plot_reliability_curve()`. Now that any model can be compared to the benchmark, the next goal is to update `ensemble.py`. 

##### 08/10/2021 Update
Updated `ensemble.py` by including a constrained linear combination weighting scheme. Still need to debug the unconstrained linear combination weighting scheme. Also added a file `ens_test.py` to test the functionality of `ensemble.py`. Still need to debug the unconstrained linear combination weighting scheme, as well as some other things.

##### 08/11/2021 Update
Included  `opening_realtime.py`, `realtime_test.py`, and `rpss.py`. The first two files open realtime forecasts and events, and test this opening procedure, respectively, while the third contains the code to calculate the rolling probability skill score. Updated `ens_test.py` to include code to create the plots that will be included in my thesis (I will include any additional plots I may produce.) Updated `metric_utils.py` to calcuate equitable threat score (ETS) and Appleman's skill score (ApSS). Tidied up and commented  every file.
 
##### January 2022 Update
Final updates on each file. Changed some of the names for the different weighting schemes in `ensemble.py`, included the rest of the figures for my thesis in `ens_test.py`, and tidied up the rest of the files.

###### References 
* Leka, K.D., Park, S.H., Kusano, K., Andries, J., Barnes, G., Bingham, S., Bloomfield, D.S., McCloskey, A.E., Delouille, V., Falconer, D. and Gallagher, P.T., 2019. A comparison of flare forecasting methods. II. Benchmarks, metrics, and performance results for operational solar flare forecasting systems. The Astrophysical Journal Supplement Series, 243(2), p.36.

* Leka, K. D.; Park, Sung-Hong, 2019, "A Comparison of Flare Forecasting Methods II: Data and Supporting Code", [https://doi.org/10.7910/DVN/HYP74O], Harvard Dataverse, V1, UNF:6:yz1noMojlzL7SZM+9flXhQ== [fileUNF]

* Wheatland, M.S., 2005. A statistical solar flare forecast method. Space Weather, 3(7).

* `metric_utils.py` obtained from https://github.com/hayesla/flare_forecast_proj/blob/main/forecast_tests/metric_utils.py

![Reliability Diagrams for each ensemble](https://user-images.githubusercontent.com/83065792/190096213-1cc2d886-6239-48d3-9e4b-526394fa96ed.jpg)

![ROC curves for each ensemble](https://user-images.githubusercontent.com/83065792/190096235-b7aeaf59-4242-4c82-8b2c-f2cc692dff4f.jpg)

![Real time RPSS for each ensemble](https://user-images.githubusercontent.com/83065792/190096278-c2e579c5-13bf-4eb4-85c3-c17e1d0bbd4a.jpg)
