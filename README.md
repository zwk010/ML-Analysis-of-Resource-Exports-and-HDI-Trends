# Natural Resource Exports & HDI Analysis

## Project Overview
<table>
  <tr>
    <td><img src="images/regressions/mineral_double_nonlinear.png" width="400"/></td>
    <td><img src="images/heatmaps/oil_heatmap.png" width="350"/></td>
  </tr>
</table>
This project investigates the complex relationship between natural resource exports and a country's Human Development Index (HDI). The impact of such exports has long been debated, with some theories suggesting accelerated development through capital inflow, while others point to negative effects like the "resource curse" and "Dutch disease," fostering corruption and complacency. We used technical methodologies such as K-means clustering, nonlinear regression, and simple neural networks to identify correlations and trends amongst data.
 Our goal is to analyze these competing hypotheses by examining the net effect of natural resource exports on the HDI, providing a unified perspective on their true impact.

We specifically focus on the export of cereal, oil, lumber, rare earth metals, and ores from countries worldwide during the period 2010-2020. The Human Development Index (HDI) serves as our primary metric, a composite index encompassing life expectancy, education, and standard of living.

![3D_all](images/all_scatter_3D.png)

# Short Report | [Full Report](Final_Report.docx)
We find that low HDI < 0.75 and high mineral (oil) resource exports > 1000 to exemplify the "resource curse" phenomenon.
Despite such high exports in excess of 1000 dollars per capita, many of these countries seem to lack improvements in HDI and have high levels of curruption or inequity.
HDI-to-export ratio in the HDI filtering function of metrics.py was not found to be useful.
![Oil_clusterings](images/regressions/mineral_double_nonlinear.png)

[Cluster 1 Resource Curse States](data/clustering_results/mineral_group_1.csv)

Angola (2013, 2014, 2015, 2017, 2018, 2019)  
Congo (2013, 2018)  
Gabon (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)  
Iraq (2013, 2014, 2015, 2016)  

[Cluster 4 Rentier States](data/clustering_results/mineral_group_4.csv) 

Algeria (2013, 2014)  
Azerbaijan (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)
Belarus (2013, 2014)  
Guyana (2020)  
Kazakhstan (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)  
Libya (2016, 2017, 2018, 2019)  
Malaysia (2013, 2014, 2015, 2017, 2018, 2019)  
Mongolia (2018, 2019)  
Oman (2015, 2020)  
Russian Federation (2013, 2014, 2015, 2016, 2017)  
Seychelles (2013, 2014, 2015, 2016, 2017, 2018, 2019)  
Trinidad and Tobago (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)  
Venezuela (2013)

[Cluster 0 Gulf States](data/clustering_results/mineral_group_0.csv)

Bahrain (2013, 2014)  
Brunei Darussalam (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020)  
Kuwait (2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020  
Oman (2013, 2014)  
Qatar (2015, 2016, 2017, 2018, 2019, 2020)  
Saudi Arabia (2013, 2014)  
United Arab Emirates (2013, 2014)   

Rare earth metals were the exception to any trend, with a resounding *negative* correlation between HDI and export levels, suggesting that exporting rare earth metals was something more developed countries quickly abandoned (enviromental damage, for example), or something forced upon lesser developed countries by necessity (pollution haven hypothesis (PHH)).
Cereals, wood, and ores had normal trends where wealth and HDI were heavily correlated and abnormalities limited.

<table>
  <tr>
    <td><img src="images/regressions/wood_double_nonlinear.png" width="300" /></td>
    <td><img src="images/regressions/ore_double_nonlinear.png" width="300" /></td>
  </tr>
  <tr>
    <td><img src="images/regressions/cereal_double_linear.png" width="300" /></td>
    <td><img src="images/regressions/inorganic_double_nonlinear.png" width="300" /></td>
  </tr>
</table>

## Key Features

* **Comprehensive Data Analysis:** Utilizes data from UN Comtrade and UNDP's HDI database for a robust analysis.
* **Multi-Methodological Approach:** Employs a combination of linear/nonlinear regression, K-means clustering, and neural networks to uncover relationships.
* **Statistical Validation:** R-squared measures are used to assess the accuracy of our regression models.
* **Visualizations:** Preliminary plots and scatterplots help in understanding global HDI trends and the relationship between specific resources and HDI.

## Analysis Workflow

1.  **Preliminary Visualizations:**
    * Plots showcasing worldwide HDI trends.
    * Scatterplots illustrating the relationship between different resource types and HDI.
    * Heatmaps for every resource type.
2.  **Regression | Neural Network:**
    * Fits regression lines or a simple neural network for each resource type, scaled for population, against HDI across all countries over the 2010-2020 period.
    * Calculates R-squared values for model accuracy assessment.
3.  **K-means Clustering:**
    * Finds dynamic number of clusters, optionally automatically using the silhouette technique, per resource type.
    * Exports 35+ rich `.csv` files for detailed cluster analysis.
    * Saves K-means models for future usage.

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/P1.git](https://github.com/yourusername/P1.git)
    cd P1
    ```
    *(Replace `yourusername` and `P1`)*

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Acquisition

The datasets used in this project are publicly available:

1.  **Resource Data, UN Comtrade:**
    * **Main portal:** [https://comtradeplus.un.org](https://comtradeplus.un.org) *Data from 2010 for all countries, individually for cereals, lumber, rare earth (inorganic) metals, and ores.*

2.  **HDI Data, UNDP:**
    * **Direct download:** [https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv](https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv)

3.  **Population Data, World Bank:**
    * **Direct download:** [https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv](https://hdr.undp.org/sites/default/files/2023-24_HDR/HDR23-24_Composite_indices_complete_time_series.csv)

### Project Structure

P1/  
├── archive/  
├── data/  
│   ├── auto_clustering/  
│   ├── clustering_results/  
│   ├── clustering_results_2/  
│   ├── Exports Data Comb/  
│   ├── Exports Per Capita/   
│   └── Exports Pop Comb/  
├── images/  
│   ├── auto_clustering_images/  
│   ├── heatmaps/  
│   ├── regressions/  
│   └── old/  
├── .gitignore  
├── cereal_clusreg.py  
├── inorganic_clusreg.py  
├── mineral_clusreg.py  
├── ore_clusreg.py  
├── wood_clusreg.py  
├── all_clusreg.py  
├── metrics.py  
├── population_cleaning.py  
├── project_1v2_0.py  
├── project_1v3.py  
├── README.md  
└── requirements.txt   


## Usage

Step 1 is data cleaning explanation- for execution skip to step 2.
1.  **Organize and preprocess data:**
    * Data is already pre-cleaned and ready, `data/Exports Pop Comb`
    * The provided HDI and resource export data was manually combined.
    * Afterwards, that data was combined with population data with:
        ```bash
        python src/population_cleaning.py
        ```
      for per capita data.
2.  **Run analysis and generate visualizations:**
    * Execute individual scripts to generate plots for regressions overlaid on clustering.
        ```bash
      python src/mineral_clusreg.py
      python src/inorganic_clusreg.py
      python src/ore_clusreg.py
      python src/wood_clusreg.py
      python src/cereals_clusreg.py
        ```
      or
      ```bash
      python src/all_clusreg.py
        ```
    * The main file can also be executed for more basic plots.
        ```bash
        python src/project_1v2_0.py
        ```
      or
      ```bash
      python src/project_1v3.py
        ```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Torch
* Matplotlib
* Seaborn
* Microsoft Excel
* Jupyter / Google Colab (legacy)

## Authors

* **Richard Cai** (rjc432) ***v2.0***
* **Wenkai Zhao** (wz459)

## Acknowledgments

* [UN Comtrade](https://comtradeplus.un.org) for trade data.
* [United Nations Development Programme (UNDP)](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI) for Human Development Index data.
* [World Bank DataBank](https://databank.worldbank.org/source/population-estimates-and-projections#) for population data.
* Wikipedia for definitions of [resource curse](https://wikipedia.org/wiki/resource_curse) and [Dutch disease](https://wikipedia.org/wiki/dutch_disease).
