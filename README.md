# CIS 419/519

This repository contains homework assignment work for CIS419/519 at UPenn in Fall 2017.  Each of the folders contain work for the assignments.  

The class website for CIS419 is located here: <http://www.seas.upenn.edu/~cis519/fall2017/>

### Google Python Class

As recommended on the CIS419 course webpage, included in the `google-python-exercises` folder is an unzip of files to work through the [Google Python Class](https://developers.google.com/edu/python/).  This was useful for learning basic syntax and expressions for Python. There are videos of the class, but they are ~ 1 hour each and it's easier to just read the material.

### Assignment 1

#### Installation

Required software for the first assignment can be found here:

- Python 2.7.x <https://www.python.org/downloads/>
- scikit-learn <http://scikit-learn.org/stable/>
- numpy <http://www.numpy.org/>
- scipy <http://www.scipy.org/>

On my Win10 x64 machine the easiest thing to do was install [Anaconda](https://www.anaconda.com/distribution/), which includes Pythin 2.7 as well as SciPy and Numpy.  Then at the Anaconda Prompt (Run as Administrator), I entered the command on the `scikit-learn` [install page](http://scikit-learn.org/stable/install.html): 

```
conda install scikit-learn
```

I then copied the text of Eric's test code, as follows, at the interpreter:

```
from sklearn import tree

X = [[0,0], [2,2]]
y = [0.5, 2.5]

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y)
clf.predict([[1,1]])
``` 
