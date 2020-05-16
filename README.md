#  Write A Data Science Blog Post - [Udacity DSND](https://www.udacity.com/course/data-scientist-nanodegree--nd025)


## Table of Contents
1. [About the Project](#about_the_project)
    1. [Goal](#goal)
    2. [How to achieve](#how_to_achieve)
2. [How to Use](#how_to_use)
    1. [Dependency](#dependency)
    2. [Installation](#installation)
    3. [Run](#run)
3. [File Structure](#file_structure)


<a name="about_the_project"></a>
## About the Project
<a name="goal"></a>
### Goal
This is one of 2nd term projects of Data Science Nanodegree Program by Udacity. The goal of the project is to write a blog post on medium to share the result of the analysis of Boston Airbnb open data using CRISP-DM process(Cross Industry Process for Data Mining). 

<a name="how_to_achieve"></a>
### How to achieve
The analysis started with three questions as follows.
1. How is the price of Airbnb in Boston adjusted in terms of seasons and locations?
2. What are the relevant features of the groups over the 80 percentile and under the 20 percentile?
3. What are the main differences between the subgroups for 2 people travelers and over 4 people travelers in the 80 percentile group?

To answer the questions, a Boston Airbnb dataset was cleaned by dropping irrelevant features, imputing Nan data, changing from categorical values to numerical values to be ready to be fed in a model based on [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) or be transformed by [PCA(Principal component analysis)](https://en.wikipedia.org/wiki/Principal_component_analysis).


<a name="how_to_use"></a>
## How to Use

<a name="dependency"></a>
### Dependency
The code should run with no issues using Python versions 3.* with the libararies as follows.
- Numpy, Pandas, Matplotlib, Seaborn, time, scipy, sklearn
- Custom library 'functions_for_AirBnB' to see a main notebook easier.

<a name="installation"></a>
### Installation
Clone the repositor below.
`https://github.com/dalpengholic/Udacity_Boston-AirBNB-Data.git`

<a name="run"></a>
### Run
1. Open a terminal and input `jupyter notebook` in the project's root directory. Then, open `p4_Boston_AirBnB_with CRISP-DM v1.ipynb` notebook.
2. Run all the cells.
      
<a name="file_structure"></a>
## File Structure
```
├── data
│   ├── calendar.csv
│   ├── listings.csv
│   ├── reviews.csv
├── README.md
├── functions_for_AirBnB.py
├── p4_Boston_AirBnB_with CRISP-DM v1.ipynb

```
