\# Specific Heat Load Profiles for Residential Consumers in District Heating Networks



High-resolution, context-aware load profiles for district heating customers based on a gradient boosting approach, considering two building types and three construction year categories.



\## Overview



This repository contains aggregated time series and engineered feature datasets for district heating customers from Aalborg.



\### Data sources



\- \*\*Smart heat meter consumption \& building context metadata\*\*:  

&nbsp; Schaffer, M., Tvedebrink, T., \& Marszal-Pomianowska, A. \*Three years of hourly data from 3021 smart heat meters installed in Danish residential buildings\*. Scientific Data, 9, 420 (2022).  

&nbsp; Data published under \*\*CC BY 4.0\*\* (http://creativecommons.org/licenses/by/4.0/). DOI: 10.5281/zenodo.6563114



\- \*\*Outdoor air temperature (hourly)\*\*:  

&nbsp; Danish Meteorological Institute (DMI) Open Data.  

&nbsp; Data published under \*\*CC BY 4.0\*\* (http://creativecommons.org/licenses/by/4.0/).  

&nbsp; Documentation: https://opendatadocs.dmi.govcloud.dk/en/DMIOpenData



The key idea is to provide load profiles \*\*transferable\*\* by conditioning them on \*\*building context metadata\*\*:



\- \*\*Building type\*\*

\- \*\*Construction year class\*\*

\- \*\*Annual consumption bands\*\* (type- \& age-aware percentiles) to avoid arbitrary cut-offs and keep bins interpretable.



---



\## Generated datasets



We export datasets at different aggregation levels:



1\. \*\*All customers (full dataset)\*\*  

&nbsp;  Includes SFH, TH and additional building types (all available customers).



2\. \*\*By building type\*\*

&nbsp;  - Single-family houses (\*\*SFH\*\*)

&nbsp;  - Terraced houses (\*\*TH\*\*)



3\. \*\*By building type + construction year class\*\*

&nbsp;  - \*\*≤ 1960\*\*

&nbsp;  - \*\*1961–1998\*\*

&nbsp;  - \*\*≥ 1999\*\*



4\. \*\*By building type + annual consumption band\*\*  

&nbsp;  Thresholds for consumption bands are defined by type-specific percentiles of annual consumption.



\### Consumption percentiles (annual consumption in MWh)



| Building type | P25  | P50  | P75  |

|---|---:|---:|---:|

| SFH | 14.15 | 17.97 | 22.18 |

| TH  | 6.65  | 8.91  | 12.25 |



\### Figures



!\[Single-family houses: annual consumption and construction year distributions](figures/annual\_consumption\_SFH.png)



!\[Terraced houses: annual consumption and construction year distributions](figures/annual\_consumption\_TH.png)



> Each exported dataset exists as:

> - an \*\*aggregated consumption time series\*\* (hourly)

> - a \*\*feature dataset\*\* derived from it (calendar + temperature features, etc.)



---



\## File naming convention



Each file encodes the selection criteria.



Example:



```text

aggregated\_\_consumertype\_single\_family\_house\_\_consumption\_14p5-17p97MWh\_\_constructionyear\_≤1960\_\_cluster\_all\_\_n0216\_\_features\_\_temp\_hourly.csv





