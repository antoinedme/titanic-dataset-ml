# Can A.I. save Jack from the Titanic?

Author: *Antoine de Marassé* https://www.linkedin.com/in/hiantoine/

Based on: Microsoft Azure (data science masterclass), Kaggle, Pandas, Sci-kit Learn and Seaborn

This notebook is a simple example where I incorporate both historical and fictionalized aspects from the 1997 epic romance and disaster movie directed by James Cameron, starring Leonardo DiCaprio and Kate Winslet. The RMS Titanic dataset is quite popular online and is a typical dataset used to make 'computer' understand patterns: Machine Learning. 

![Opening Antoine de Marasse](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/opening-image.png)   

Contents:
- [History of the RMS Titanic passenger liner](https://github.com/antoinedme/titanic-dataset-ml#history-of-the-rms-titanic-passenger-liner)
- [Exploring the Passengers dataset entries](https://github.com/antoinedme/titanic-dataset-ml#exploring-the-passengers-dataset-entries)
- [Analysing the data](https://github.com/antoinedme/titanic-dataset-ml#analysing-the-data) (more having a look and trying to analyse correlation of parameters and to plot simple stuffs)
- [Cleaning the data](https://github.com/antoinedme/titanic-dataset-ml#cleaning-the-data) (basic cleaning, no real process here)
- [Splitting the dataset for training and testing](https://github.com/antoinedme/titanic-dataset-ml#splitting-the-dataset-for-training-and-testing)
- [Applying Logistic Regression](https://github.com/antoinedme/titanic-dataset-ml#applying-logistic-regression)
- [Evaluating the model](https://github.com/antoinedme/titanic-dataset-ml#evaluating-the-model)
- [What about Jack and Rose?](https://github.com/antoinedme/titanic-dataset-ml#what-about-jack-and-rose)
- [Applying Decision Tree](https://github.com/antoinedme/titanic-dataset-ml#applying-decision-tree)
- [Final remarks](https://github.com/antoinedme/titanic-dataset-ml#final-remarks) (wait, there's a surprise here!)

Directly access all the code on the Jupyter Notebook (Python): 

https://github.com/antoinedme/titanic-dataset-ml/blob/master/Titanic-MachineLearning.ipynb

The deck of slides: 
https://drive.google.com/file/d/1IdNF3zSMvRj5gbuYC7UgiBS8uOOYoeAH/view?usp=sharing

Let's get started!


## History of the RMS Titanic passenger liner

![Image of Titanic](https://titanichistoricalsociety.org/wp-content/uploads/2017/09/titanic_historical_society_homepage_harley_crossley.jpg?bd15df&bd15df)   

RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after striking an iceberg during her voyage from Southampton to New York City. 
Of the estimated 2,224 passengers and crew aboard, **more than 1,500 died**, making the sinking one of modern history's deadliest peacetime commercial marine disasters. 
A disproportionate number of men were left aboard because of a "women and children first" protocol for loading lifeboats. At 2:20 a.m., she broke apart and foundered with well over one thousand people still aboard. Just under two hours after Titanic sank, the Cunard liner RMS Carpathia arrived and brought aboard an estimated 705 survivors. 

## Exploring the Passengers dataset entries

This dataset describes the survival status of individual passengers on the Titanic. 

![Passengers table](https://user-images.strikinglycdn.com/res/hrscywv4p/image/upload/c_limit,f_auto,h_2000,q_90,w_1200/107158/Screen_Shot_2015-08-03_at_1.57.45_AM_ibp1u8.png)   

The dataset has 1309 entries accross 10 variables:
- `survived`: 0 = No, 1 = Yes. (As we can see on the table above `survived` mean, 38,19% of passengers survived)
- `pclass`: Ticket category from first to third class. 
- `fare`: Passenger fare (First class fare: 87.50, Second Class: 21.17, Third Class: 13.29)
- `ticket`: Passenger ticket number
- Demographics: `Sex` (466 females and 843 males), `Age` (Median age of 27 years old)
- `sibsp`, `parch`: Number of siblings or spouses aboard, number of parents or children aboard
- `cabin`: Cabin number, `embarked`: Port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton

*Note: In 1912, skilled shipyard workers who built Titanic earned £2 ($10) per week. Unskilled workers earned £1 or less per week. A single First Class berth would cost these workers 4 to 8 months wages. $100 in 1912 → $2,667 in 2020*

## Analysing the data

We will start by grouping the survival data per sex and class. To do so we will use the basic `groupby` function on our whole dataset, and use `seaborn`, the Python data visualization library based on matplotlib, that actually already uses this specific dataset in its `catplot` example:

```
exploratory.groupby(['sex_is_male','pclass'])['survived'].mean()
graph = sns.catplot(x="sex_is_male", y="survived", hue="pclass", kind="bar", palette="muted", data=exploratory)
```

Survival probability: for women on 1st class is: 96,5% compared to men only 34,1% When we look at the 3rd class, the probability drops to 49,1% for women and 15,2% for men. 

Women and children first!

![Survival rate accross classes per sex](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/survival-analysis.png)


## Cleaning the data

In this notebook, we won't use some of the variables such as `home.dest`, `embarked` or `cabin`. To do so we will drop:

```
titanic.drop(['name','body','boat','cabin','ticket','embarked','home.dest'],axis=1,inplace=True)
```

Then we will fill in the missing variables in `fare` (1 missing) and `age`(263 missing) using the median values:

```
titanic['fare'] = titanic.groupby('pclass')['fare'].apply(lambda x: x.fillna(x.median()))
titanic['age'] = titanic.groupby('sex')['age'].apply(lambda x: x.fillna(x.median()))
```

## Splitting the dataset for training and testing

For the following, I will use Scikit-Learn:
Install the scikit-learn - Machine Learning in Python: `conda install -c intel scikit-learn`
- Simple and efficient tools for predictive data analysis
- Built on NumPy, SciPy, and matplotlib Link: https://scikit-learn.org/stable/

The library: `from sklearn.model_selection import train_test_split`

We first prepare our dataset into two variables `X` and `y`.The `X` variable is the whole dataframe without the `survived` parameter. That means all passengers variables are there, without knowing if they died or survived the tragedy. The `y` is the target `survived` 0/1 we wan't to know. 

```
X = iceberg.drop(['survived'],axis=1)
y = iceberg['survived']
```

We will now split our data (can be: pandas dataframe, numpy arrays, lists) into random train and test subsets. For this function we need to define few parameters: `random_state`, the seed used by the random number generator, and `test_size`, the proportion of the dataset to include in the test split, here 30% of the total dataset of 1309 entries:

```
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=67)
```

## Applying Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. 

In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). 

Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail which is represented by an indicator variable, where the two values are labelled "0" and "1". Here it will depicts survived/died.

We start by the sklearn imports that enables us to create such model:

```
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
```
The steps are now:
- Implement regularized logistic regression
- Fit the model according to the given training data.
- Predict class labels for samples in X.
```
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
```

![Logistic regression example](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/regression-illus.png)

The `predictions` is: array([0, …, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])

Now the model is implemented and trained, we can switch to the next part.

## Evaluating the model

In contrast to linear regression, logistic regression does not produce an 𝑅2 score by which we can assess the accuracy of our model. In order to evaluate that, we will use a classification report, a confusion matrix, and the accuracy score. We import the related tools from sklearn.metrics:

```
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```
In logistic regression, the response variable describes the probability that the outcome is the positive case. If the response variable is equal to or exceeds a discrimination threshold, the positive class is predicted; otherwise, the negative class is predicted. The response variable is modelled as a function of a linear combination of the explanatory variables using the logistic function. Given by the following equation, the logistic function always returns a value between zero and one: 

```
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))
```
Output: 0.7582697201017812. The accuracy score is 76%.

## What about Jack and Rose?

In 1912 Southampton, 17-year-old first-class passenger Rose DeWitt Bukater, her fiancé Cal Hockley, and her mother Ruth board the luxurious Titanic. Ruth emphasizes that Rose's marriage will resolve their family's financial problems and allow them to retain their upper-class status. Distraught over the engagement, Rose climbs over the stern and contemplates suicide; Jack Dawson, a penniless artist, intervenes and discourages her. Discovered with Jack, Rose tells a concerned Cal that she was peering over the edge and Jack saved her from falling. 

Jack and Rose develop a tentative friendship, despite Cal and Ruth being wary of him. Following dinner, Rose secretly joins Jack at a party in third class. 

![Jack attributes](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/couple.png)


Let's create the data for our lovely couple (we will follow the structure: class, age, parents on board, fare, sex): Jack is on third class, around 27 years old, alone on board, only paid 8$ and is a male `jack = [3, 27, 0, 8,  1]`, and rose is on first class, around 22 years old, with family and paid 60$ `rose = [1, 22, 1, 60, 0]`.

```
survival = lr.predict(pd.DataFrame(np.array([jack, rose]), columns=['pclass', 'age', 'parch', 'fare','sex_is_male']))
```

After braving several obstacles, Jack and Rose return to the boat deck. The lifeboats have departed and passengers are falling to their deaths as the stern rises out of the water. The ship breaks in half, dropping the stern into the water. Jack and Rose ride it into the ocean and he helps her onto a wooden panel buoyant enough for only one person. He assures her that she will die an old woman, warm in her bed. 

Jack dies of hypothermia but Rose is saved. 

And the results from predictions are:

![Results](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/results.png)

Wait is there something more we can do?

## Applying Decision Tree

We will now still try to save Jack by discussing a simple, nonlinear model for classification and regression tasks: the decision tree.

```
from sklearn import tree
tr = tree.DecisionTreeClassifier()
tr.fit(X_train, y_train)
tr_predictions = tr.predict(X_test)
print(accuracy_score(y_test,tr_predictions))
```

![Jack attributes](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/decision-tree-illustration.png)


Output: 0.7150127226463104. The accuracy score is 71,5%.

Let's try this now with Jack and Rose data.

And the results from predictions are:

![Results](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/results.png)

Jack dies again!


## Final remarks 

At 11:40 p.m. (ship's time) on 14 April, lookout Frederick Fleet spotted an iceberg immediately ahead of Titanic and alerted the bridge. First Officer William Murdoch ordered the ship to be steered around the obstacle and the engines to be stopped, but it was too late; the starboard side of Titanic struck the iceberg, creating a series of holes below the waterline. It soon became clear that the ship was doomed, as she could not survive more than four compartments being flooded. Titanic began sinking bow-first, with water spilling from compartment to compartment as her angle in the water became steeper. Third-class passengers were largely left to fend for themselves, causing many of them to become trapped below decks as the ship filled with water. The "women and children first" protocol was generally followed when loading the lifeboats, and most of the male passengers and crew were left aboard. 

It looks like both `logistic regression` and `decision tree` methods can't save our Jack! The vertical stern of the ship plunges down shrieking and groaning, with bodies falling hundreds of feet down toward churning water. Some fans will never let go of the possibility that there was room enough for both Jack and Rose on that door at the end of Titanic. Director James Cameron has an explanation for them that doesn’t involve physics, but rather art. “Had he lived, the ending of the film would have been meaningless,” he said in a recent Vanity Fair interview. “The film is about death and separation; he had to die.”

Unless...

Click this link for an alternate Titanic ending: https://www.youtube.com/watch?v=iphqRPaaeP8

![Ending Video](https://raw.githubusercontent.com/antoinedme/titanic-dataset-ml/master/ressources/img/youtube-video.png)






thank you very much, 

by *Antoine de Marassé* https://www.linkedin.com/in/hiantoine/





Access all the code on the Jupyter Notebook (Python): 

https://github.com/antoinedme/titanic-dataset-ml/blob/master/Titanic-MachineLearning.ipynb


