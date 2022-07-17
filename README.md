# Predicting the class of forest cover

This repository covers the analyses for predicting the class of forest cover (the predominant kind of tree cover) from strictly cartographic and environmental variables.

The actual forest cover type for a given observation (30 x 30-meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains categorical data for qualitative independent variables (wilderness areas and soil types).

Further details on the data could be found in the next link https://archive.ics.uci.edu/ml/datasets/Covertype

This notebook shows the steps followed in order to predict the cover type using the Logistic Regression models for two differents kinds a of classification task, a binary task in which I want to predict cover type 7 - Krummholz, and a multiclass classification taks in which I want to predict the forest cover kind that is predominant given the cartographic and enviromental variables.
