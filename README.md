# :passenger_ship: Titanic dataset discovery
Visit the notebook here: https://github.com/antoinedme/titanic-dataset-ml/blob/master/Titanic-MachineLearning.ipynb


## History of the RMS Titanic passenger liner
![Image of Titanic](https://titanichistoricalsociety.org/wp-content/uploads/2017/09/titanic_historical_society_homepage_harley_crossley.jpg?bd15df&bd15df)   
RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after striking an iceberg during her voyage from Southampton to New York City. 
Of the estimated 2,224 passengers and crew aboard, **more than 1,500 died**, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 
A disproportionate number of men were left aboard because of a "women and children first" protocol for loading lifeboats. At 2:20 a.m., she broke apart and foundered with well over one thousand people still aboard. Just under two hours after Titanic sank, the Cunard liner RMS Carpathia arrived and brought aboard an estimated 705 survivors. 

## Exploring the Passengers dataset entries
This dataset describes the survival status of individual passengers on the Titanic, it has 1309 entries accross 10 variables:
- `survived`: 0 = No, 1 = Yes. **(As we can see on the table above `survived` mean, 38,19% of passengers survived)**

- `pclass`: Ticket category from first to third class. **(First class fare: 87.50, Second Class: 21.17, Third Class: 13.29)**
- `fare`: Passenger fare
- `ticket`: Passenger ticket number

- Demographics: `Sex`, `Age`
- `sibsp`, `parch`: Number of siblings or spouses aboard, number of parents or children aboard

- `cabin`: Cabin number, `embarked`: Port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton


In 1912, skilled shipyard workers who built Titanic earned £2 ($10) per week. Unskilled workers earned £1 or less per week. A single First Class berth would cost these workers 4 to 8 months wages. $100 in 1912 → $2,667 in 2020

![Passengers table](https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,f_auto,h_2000,q_90,w_1200/107158/Screen_Shot_2015-08-03_at_1.57.45_AM_ibp1u8.png)   

## Analysing the data

Survival probability: for women on 1st class is: 96,5% compared to men only 34,1% When we look at the 3rd class, the probability drops to 49,1% for women and 15,2% for men. 

`exploratory.groupby(['sex_is_male','pclass'])['survived'].mean()`

`graph = sns.catplot(x="sex_is_male", y="survived", hue="pclass", kind="bar", palette="muted", data=exploratory)`

![Survival rate accross classes per sex](https://seaborn.pydata.org/_images/categorical_36_0.png)
Women and children first!

## Cleaning the data
In this notebook, we won't use some of the variables such as `home.dest`, `embarked` or `cabin`. To do so we will drop:

`titanic.drop(['name','body','boat','cabin','ticket','embarked','home.dest'],axis=1,inplace=True)`

Then we will fill in the missing variables in `fare` (1 missing) and `age`(263 missing) using the median values:

`titanic['fare'] = titanic.groupby('pclass')['fare'].apply(lambda x: x.fillna(x.median()))`

`titanic['age'] = titanic.groupby('sex')['age'].apply(lambda x: x.fillna(x.median()))`

## Split the dataset for training and testing

Visit the notebook here: https://github.com/antoinedme/titanic-dataset-ml/blob/master/Titanic-MachineLearning.ipynb

thank you very much, 
by Antoine de Marassé
