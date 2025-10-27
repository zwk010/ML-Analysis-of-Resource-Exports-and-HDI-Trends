# ML-Corruption-and-Expenditure

A forked project from [ML Resource Analysis](https://github.com/zwk010/ML-Analysis-of-Resource-Exports-and-HDI-Trends) focusing more directly on corruption, seeking to understand if certain types of government spending is correlated with corruption. Even the most disreuputable governments in the digital world today publish at least some high-level spending data, from which many inconsistencies and outliers can already be seen. This project aims to develop a classifier to categorize a government's corruption risk based on rudimentary reported statistics.  

## Data Sources:  
Open-source data was obtained from the Quality of Government (QoG) Institute affiliated with the University of Gothenburg.
[QOG Dataset](https://www.gu.se/en/quality-government/qog-data)  
[Specific Subset of Data](https://datafinder.qog.gu.se/downloads?download=gfs_def,gfs_ecaf,gfs_educ,gfs_envr,gfs_gps,gfs_hca,gfs_heal,gfs_pos,gfs_rcr,gfs_sp,ictd_revinsc,bci_bci)  

## Decision Tree:  
Filtered rows total: 1341  
Train samples: 1072, Test samples: 269  
R^2: 0.6609, MSE: 139.6762, MAE: 8.5919  
Features were continuous ratios of government sector spending to total government revenue from 1972 to 2022. Likewise, labels were continuous BCI indicies (from 0.0 to 1.0) of percieved corruption. Years with incomplete or missing spending data were skipped, but countries had their years analyzed independently so complete chronological records were not required. Over 200 countries' data was polled.  
*Feature importances:*  
safety_ratio         0.506031  
environment_ratio    0.142394  
recreation_ratio     0.108229  
social_ratio         0.054914  
housing_ratio        0.052054  
health_ratio         0.045323  
economic_ratio       0.043523  
education_ratio      0.022957  
services_ratio       0.019592  
defense_ratio        0.004984  
It appears that the raio of safety (internal security) spending to revenue was strongly correlated and rather important in the consideration of BCI (Bayesian Corruption Index). This would make sense as repressive regimes tend to be corrupt from the lack of transparency, and thus spend much covering up instabilities arising from their inefficiencies and shortcomings. Most remaining features, with the exception of defense, fell into a similar archetype of minimal correlation to corruption. These mostly were focused on development and care of citizens, and would be relatively similar in terms of corruption exploitation potential. The other outlier, defense to revenue ratio, had no correlation or importance in the decision tree whatsoever. The reason is probably twofold: defense spending is often spurned by conflict, during which revenue streams are unlikely to rise (unlike total expenditures), and threats to a country that would require defensive preperations often are not directly correlated to corruption.


![image_1](images/corruption/decision_tree.png)  

## Random Forest:  
![3D_all](images/corruption/random_forest_importances.png)  
![3D_all](images/corruption/random_forest_test.png)
