\# Specific Heat Load Profiles for Residential Consumers in District Heating Networks



High-resolution, context-aware load profiles for district heating customers based on a gradient boosting approach, considering two building types and three construction year categories.



\## Overview



This repository contains aggregated time series and engineered feature datasets for district heating customers from Aalborg.



\*\*Data sources\*\*

\- \*\*Smart heat meter consumption \& building context metadata\*\*: Schaffer, M., Tvedebrink, T., \& Marszal-Pomianowska, A. \*Three years of hourly data from 3021 smart heat meters installed in Danish residential buildings\*. Scientific Data, 9, 420 (2022). Data published under \*\*CC BY 4.0\*\* (http://creativecommons.org/licenses/by/4.0/). DOI: 10.5281/zenodo.6563114

\- \*\*Outdoor air temperature (hourly)\*\*: Danish Meteorological Institute (DMI) Open Data. Data published under \*\*CC BY 4.0\*\* (http://creativecommons.org/licenses/by/4.0/). Documentation: https://opendatadocs.dmi.govcloud.dk/en/DMIOpenData



The key idea is to provide load profiles \*\*transferable\*\* by conditioning them on \*\*building context metadata\*\*:

\- \*\*Building type\*\*

\- \*\*Construction year class\*\*

\- \*\*Annual consumption bands\*\* (type- \& age-aware percentiles)  

&nbsp; to avoid arbitrary cut-offs and keep bins interpretable.



The produced datasets can be used to train and evaluate models that generalize better to new customer groups where building type and approximate construction year are typically known.



---



\## Generated feature datasets



We export datasets at different aggregation levels:



1\) \*\*All customers (full dataset)\*\*  

&nbsp;  Includes SFH, TH and additional building types (all available customers).



2\) \*\*By building type\*\*

\- Single-family houses (\*\*SFH\*\*)

\- Terraced houses (\*\*TH\*\*)



3\) \*\*By building type + construction year class\*\*  

&nbsp;  For each of SFH and TH, we segment the Danish building stock into three cohorts reflecting Denmark's thermal insulation regulation milestones. These correspond to major steps in tightening U-value limits / insulation standards.

\- \*\*≤ 1960\*\*

\- \*\*1961–1998\*\*

\- \*\*≥ 1999\*\*



4\) \*\*By building type + construction year class + annual consumption band\*\*

\- Thresholds for consumption bands are defined by \*\*type-specific percentiles\*\* of annual consumption; global percentiles and the average of construction-year specific percentiles are very close.

\- Bands used for dataset export:



| Building type |   P25 |   P50 |   P75 |

|---|---:|---:|---:|

| SFH | 14.15 | 17.97 | 22.18 |

| TH  |  6.65 |  8.91 | 12.25 |



!\[Single-family houses: annual consumption and construction year distributions](figures/annual\_consumption\_SFH.png)  

!\[Terraced houses: annual consumption and construction year distributions](figures/annual\_consumption\_TH.png)



---



\## File naming convention



Each file encodes the selection criteria.



Example:

`aggregated\_\_consumertype\_single\_family\_house\_\_consumption\_14p5-17p97MWh\_\_constructionyear\_≤1960\_\_cluster\_all\_\_n0216\_\_features\_\_temp\_hourly.csv`



Meaning:

\- `consumertype\_single\_family\_house` → building type filter

\- `consumption\_14p5-17p97MWh` → annual consumption band (MWh); decimal values use `p` as separator

\- `constructionyear\_≤1960` → construction year class

\- `cluster\_all` → no further clustering inside the selection (single group export)

\- `n0216` → number of households/files included in this aggregated dataset

\- `features\_\_temp\_hourly` → feature dataset using hourly temperature mode





