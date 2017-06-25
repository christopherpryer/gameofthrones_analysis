

<h1>Welcome to my Game of Thrones Analysis</h1>
<hr>

<p>Below is my first Kaggle project. This project will consist of the use of the following datasets:</p>

<ul>
    <li>
        <a href="https://github.com/chrisalbon/war_of_the_five_kings_dataset">battles.csv</a> represent data related to the War of the Five Kings from George R.R. Martin's A Song Of Ice And Fire series.
    </li>
    <li>
         <a href="https://github.com/benkahle/bayesianGameofThrones">character-deaths.csv</a> is data related to a <a href="http://allendowney.blogspot.com/2015/03/bayesian-survival-analysis-for-game-of.html">Bayesian Survival Analysis</a> of Game of Thrones.
    </li>
    <li>
        <a href="https://www.kaggle.com/mylesoneill/game-of-thrones">character-predictions.csv</a> is data scraped from <a href="http://awoiaf.westeros.org/index.php/Main_Page">a wiki</a> that covers some predictions, and <a href="https://got.show/machine-learning-algorithm-predicts-death-game-of-thrones">here</a> is the methodolgy that may or may not be covered here.
    </li>
</ul>
<hr>

<h2>1: Load Data</h2>


```python
import os
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import matplotlib.pyplot as mplt

#go offline with plotly
init_notebook_mode(connected=True) 

#set working directory
WRKSPC = 'C:\\Users\\Chris\\Analytics\\gameofthrones_analysis\\'

#loading data as dataframes
battles = pd.read_csv(WRKSPC+'battles.csv')
deaths = pd.read_csv(WRKSPC+'character-deaths.csv')
predictions = pd.read_csv(WRKSPC+'character-predictions.csv')

```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>


<hr>


<h3>Battles: A Preliminary Summary</h3>
<ul>
<li>
<a href="https://github.com/christopherpryer/gameofthrones_analysis/blob/master/battles.csv">battles.csv</a> contains the name of each battle, the year it happened, who was attacking (along with a somewhat more granular level of who was involved), who consisted of the defense, house related to each side, count of deaths and major deaths, captures, sizes, region, some notes, and seasonality.
</li>
<li>
Just looking at the data the first areas I'd like to address are the significance of vital factors regarding each side's parameters.
</li>
<li>
It is also worth noting that this data set paints a higher level picture of some deaths. It is a smaller set of data but in some ways this might be worth comming back to for some relations with other data.
</li>
</ul>


```python
#print battles
battles.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>year</th>
      <th>battle_number</th>
      <th>attacker_king</th>
      <th>defender_king</th>
      <th>attacker_1</th>
      <th>attacker_2</th>
      <th>attacker_3</th>
      <th>attacker_4</th>
      <th>defender_1</th>
      <th>...</th>
      <th>major_death</th>
      <th>major_capture</th>
      <th>attacker_size</th>
      <th>defender_size</th>
      <th>attacker_commander</th>
      <th>defender_commander</th>
      <th>summer</th>
      <th>location</th>
      <th>region</th>
      <th>note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Battle of the Golden Tooth</td>
      <td>298</td>
      <td>1</td>
      <td>Joffrey/Tommen Baratheon</td>
      <td>Robb Stark</td>
      <td>Lannister</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tully</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>15000.0</td>
      <td>4000.0</td>
      <td>Jaime Lannister</td>
      <td>Clement Piper, Vance</td>
      <td>1.0</td>
      <td>Golden Tooth</td>
      <td>The Westerlands</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Battle at the Mummer's Ford</td>
      <td>298</td>
      <td>2</td>
      <td>Joffrey/Tommen Baratheon</td>
      <td>Robb Stark</td>
      <td>Lannister</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Baratheon</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>120.0</td>
      <td>Gregor Clegane</td>
      <td>Beric Dondarrion</td>
      <td>1.0</td>
      <td>Mummer's Ford</td>
      <td>The Riverlands</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Battle of Riverrun</td>
      <td>298</td>
      <td>3</td>
      <td>Joffrey/Tommen Baratheon</td>
      <td>Robb Stark</td>
      <td>Lannister</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tully</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>15000.0</td>
      <td>10000.0</td>
      <td>Jaime Lannister, Andros Brax</td>
      <td>Edmure Tully, Tytos Blackwood</td>
      <td>1.0</td>
      <td>Riverrun</td>
      <td>The Riverlands</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Battle of the Green Fork</td>
      <td>298</td>
      <td>4</td>
      <td>Robb Stark</td>
      <td>Joffrey/Tommen Baratheon</td>
      <td>Stark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Lannister</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>18000.0</td>
      <td>20000.0</td>
      <td>Roose Bolton, Wylis Manderly, Medger Cerwyn, H...</td>
      <td>Tywin Lannister, Gregor Clegane, Kevan Lannist...</td>
      <td>1.0</td>
      <td>Green Fork</td>
      <td>The Riverlands</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Battle of the Whispering Wood</td>
      <td>298</td>
      <td>5</td>
      <td>Robb Stark</td>
      <td>Joffrey/Tommen Baratheon</td>
      <td>Stark</td>
      <td>Tully</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Lannister</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1875.0</td>
      <td>6000.0</td>
      <td>Robb Stark, Brynden Tully</td>
      <td>Jaime Lannister</td>
      <td>1.0</td>
      <td>Whispering Wood</td>
      <td>The Riverlands</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
print battles.shape
```

    (38, 25)
    


```python
#print battles.describe()
battles.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>battle_number</th>
      <th>defender_3</th>
      <th>defender_4</th>
      <th>major_death</th>
      <th>major_capture</th>
      <th>attacker_size</th>
      <th>defender_size</th>
      <th>summer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38.000000</td>
      <td>38.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.000000</td>
      <td>37.000000</td>
      <td>24.000000</td>
      <td>19.000000</td>
      <td>37.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>299.105263</td>
      <td>19.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.351351</td>
      <td>0.297297</td>
      <td>9942.541667</td>
      <td>6428.157895</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.689280</td>
      <td>11.113055</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.483978</td>
      <td>0.463373</td>
      <td>20283.092065</td>
      <td>6225.182106</td>
      <td>0.463373</td>
    </tr>
    <tr>
      <th>min</th>
      <td>298.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>100.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.000000</td>
      <td>10.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1375.000000</td>
      <td>1070.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>299.000000</td>
      <td>19.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4000.000000</td>
      <td>6000.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.000000</td>
      <td>28.750000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>8250.000000</td>
      <td>10000.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>300.000000</td>
      <td>38.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100000.000000</td>
      <td>20000.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<hr>
<br>
<h3>Deaths: A Preliminary Summary</h3>
<ul>
<li>
<a href="https://github.com/christopherpryer/gameofthrones_analysis/blob/master/character-deaths.csv">character-deaths.csv</a> contains the name of the character, allegiance to what house, death year, book they died in, chapter in which they died, gender, nobility, GoT appearance, and a each book they appeared in. 
</li>
<li>
Off the bat the scale of this file will allow me to get a feel for prevelence of death. There might be relations with the frequency of death and a certain house.
</li>
<li>
There isn't too much depth, at least at first glance. The file just states when and if someone died (maybe how many times they died too -- we'll cover this later).
</li>
</ul>


```python
#print deaths
deaths.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Allegiances</th>
      <th>Death Year</th>
      <th>Book of Death</th>
      <th>Death Chapter</th>
      <th>Book Intro Chapter</th>
      <th>Gender</th>
      <th>Nobility</th>
      <th>GoT</th>
      <th>CoK</th>
      <th>SoS</th>
      <th>FfC</th>
      <th>DwD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Addam Marbrand</td>
      <td>Lannister</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>56.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aegon Frey (Jinglebell)</td>
      <td>None</td>
      <td>299.0</td>
      <td>3.0</td>
      <td>51.0</td>
      <td>49.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aegon Targaryen</td>
      <td>House Targaryen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adrack Humble</td>
      <td>House Greyjoy</td>
      <td>300.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aemon Costayne</td>
      <td>Lannister</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print deaths.shape
```

    (917, 13)
    


```python
#print deaths.describe()
deaths.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Death Year</th>
      <th>Book of Death</th>
      <th>Death Chapter</th>
      <th>Book Intro Chapter</th>
      <th>Gender</th>
      <th>Nobility</th>
      <th>GoT</th>
      <th>CoK</th>
      <th>SoS</th>
      <th>FfC</th>
      <th>DwD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>305.000000</td>
      <td>307.000000</td>
      <td>299.000000</td>
      <td>905.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
      <td>917.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>299.157377</td>
      <td>2.928339</td>
      <td>40.070234</td>
      <td>28.861878</td>
      <td>0.828790</td>
      <td>0.468920</td>
      <td>0.272628</td>
      <td>0.353326</td>
      <td>0.424209</td>
      <td>0.272628</td>
      <td>0.284624</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.703483</td>
      <td>1.326482</td>
      <td>20.470270</td>
      <td>20.165788</td>
      <td>0.376898</td>
      <td>0.499305</td>
      <td>0.445554</td>
      <td>0.478264</td>
      <td>0.494492</td>
      <td>0.445554</td>
      <td>0.451481</td>
    </tr>
    <tr>
      <th>min</th>
      <td>297.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>299.000000</td>
      <td>2.000000</td>
      <td>25.500000</td>
      <td>11.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>299.000000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>27.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.000000</td>
      <td>4.000000</td>
      <td>57.000000</td>
      <td>43.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>300.000000</td>
      <td>5.000000</td>
      <td>80.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



<hr>
<br>
<h3>Predictions: A Preliminary Summary</h3>
<ul>
<li>
<a href="https://github.com/christopherpryer/gameofthrones_analysis/blob/master/character-predictions.csv">character-predictions.csv</a> contains more interesting data. This will need more than a high level glance.
</li>
</ul>


```python
#print predictions
predictions.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No</th>
      <th>actual</th>
      <th>pred</th>
      <th>alive</th>
      <th>plod</th>
      <th>name</th>
      <th>title</th>
      <th>male</th>
      <th>culture</th>
      <th>dateOfBirth</th>
      <th>...</th>
      <th>isAliveHeir</th>
      <th>isAliveSpouse</th>
      <th>isMarried</th>
      <th>isNoble</th>
      <th>age</th>
      <th>numDeadRelations</th>
      <th>boolDeadRelations</th>
      <th>isPopular</th>
      <th>popularity</th>
      <th>isAlive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.054</td>
      <td>0.946</td>
      <td>Viserys II Targaryen</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0.605351</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0.387</td>
      <td>0.613</td>
      <td>Walder Frey</td>
      <td>Lord of the Crossing</td>
      <td>1</td>
      <td>Rivermen</td>
      <td>208.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>97.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.896321</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.493</td>
      <td>0.507</td>
      <td>Addison Hill</td>
      <td>Ser</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.267559</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.076</td>
      <td>0.924</td>
      <td>Aemma Arryn</td>
      <td>Queen</td>
      <td>0</td>
      <td>NaN</td>
      <td>82.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>23.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.183946</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0.617</td>
      <td>0.383</td>
      <td>Sylva Santagar</td>
      <td>Greenstone</td>
      <td>0</td>
      <td>Dornish</td>
      <td>276.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.043478</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
print predictions.shape
```

    (1946, 33)
    


```python
#print predictions.describe()
predictions.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.No</th>
      <th>actual</th>
      <th>pred</th>
      <th>alive</th>
      <th>plod</th>
      <th>male</th>
      <th>dateOfBirth</th>
      <th>DateoFdeath</th>
      <th>book1</th>
      <th>book2</th>
      <th>...</th>
      <th>isAliveHeir</th>
      <th>isAliveSpouse</th>
      <th>isMarried</th>
      <th>isNoble</th>
      <th>age</th>
      <th>numDeadRelations</th>
      <th>boolDeadRelations</th>
      <th>isPopular</th>
      <th>popularity</th>
      <th>isAlive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>433.000000</td>
      <td>444.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>...</td>
      <td>23.000000</td>
      <td>276.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>433.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
      <td>1946.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>973.500000</td>
      <td>0.745632</td>
      <td>0.687050</td>
      <td>0.634470</td>
      <td>0.365530</td>
      <td>0.619219</td>
      <td>1577.364896</td>
      <td>2950.193694</td>
      <td>0.198356</td>
      <td>0.374615</td>
      <td>...</td>
      <td>0.652174</td>
      <td>0.778986</td>
      <td>0.141829</td>
      <td>0.460946</td>
      <td>-1293.563510</td>
      <td>0.305755</td>
      <td>0.074512</td>
      <td>0.059096</td>
      <td>0.089584</td>
      <td>0.745632</td>
    </tr>
    <tr>
      <th>std</th>
      <td>561.906131</td>
      <td>0.435617</td>
      <td>0.463813</td>
      <td>0.312637</td>
      <td>0.312637</td>
      <td>0.485704</td>
      <td>19565.414460</td>
      <td>28192.245529</td>
      <td>0.398864</td>
      <td>0.484148</td>
      <td>...</td>
      <td>0.486985</td>
      <td>0.415684</td>
      <td>0.348965</td>
      <td>0.498601</td>
      <td>19564.340993</td>
      <td>1.383910</td>
      <td>0.262669</td>
      <td>0.235864</td>
      <td>0.160568</td>
      <td>0.435617</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-298001.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>487.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.391250</td>
      <td>0.101000</td>
      <td>0.000000</td>
      <td>240.000000</td>
      <td>282.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.013378</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>973.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.735500</td>
      <td>0.264500</td>
      <td>1.000000</td>
      <td>268.000000</td>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.033445</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1459.750000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.899000</td>
      <td>0.608750</td>
      <td>1.000000</td>
      <td>285.000000</td>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1946.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>298299.000000</td>
      <td>298299.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>15.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>



<hr>
<br>
<h1>2: Exploring the data</h1>
<hr>
<br>

<h3>Battles</h3>

<p>Let's count the number of commanders listed as the attackers and see if there is a relation between this field and the factor that is scale of a side. First I'll print the attacking commanders for each battle where attacker size is null or attacking commander is null.</p>


```python
battles_w_nulls = battles[pd.isnull(battles['attacker_size']) | pd.isnull(battles['attacker_commander'])]
battles_w_nulls[['name', 'attacker_commander']]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>attacker_commander</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Battle at the Mummer's Ford</td>
      <td>Gregor Clegane</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sack of Darry</td>
      <td>Gregor Clegane</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Battle of Moat Cailin</td>
      <td>Victarion Greyjoy</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sack of Torrhen's Square</td>
      <td>Dagmer Cleftjaw</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Siege of Darry</td>
      <td>Helman Tallhart</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Battle of the Burning Septry</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Battle of the Ruby Ford</td>
      <td>Gregor Clegane</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Retaking of Harrenhal</td>
      <td>Gregor Clegane</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Siege of Seagard</td>
      <td>Walder Frey</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Fall of Moat Cailin</td>
      <td>Ramsey Bolton</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sack of Saltpans</td>
      <td>Rorge</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Battle of the Shield Islands</td>
      <td>Euron Greyjoy, Victarion Greyjoy</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Invasion of Ryamsport, Vinetown, and Starfish ...</td>
      <td>Euron Greyjoy, Victarion Greyjoy</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Second Seige of Storm's End</td>
      <td>Mace Tyrell, Mathis Rowan</td>
    </tr>
  </tbody>
</table>
</div>



<p>These records are going to have to be left out of the analysis, so let's create a new dataframe without them</p>


```python
#assign a new df for clean battles data
battles_df = battles[battles.attacker_size.notnull() & battles.attacker_commander.notnull()]
battles_df[['name','attacker_size','attacker_commander']]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>attacker_size</th>
      <th>attacker_commander</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Battle of the Golden Tooth</td>
      <td>15000.0</td>
      <td>Jaime Lannister</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Battle of Riverrun</td>
      <td>15000.0</td>
      <td>Jaime Lannister, Andros Brax</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Battle of the Green Fork</td>
      <td>18000.0</td>
      <td>Roose Bolton, Wylis Manderly, Medger Cerwyn, H...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Battle of the Whispering Wood</td>
      <td>1875.0</td>
      <td>Robb Stark, Brynden Tully</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Battle of the Camps</td>
      <td>6000.0</td>
      <td>Robb Stark, Tytos Blackwood, Brynden Tully</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Battle of Deepwood Motte</td>
      <td>1000.0</td>
      <td>Asha Greyjoy</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Battle of the Stony Shore</td>
      <td>264.0</td>
      <td>Theon Greyjoy</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Battle of Torrhen's Square</td>
      <td>244.0</td>
      <td>Rodrik Cassel, Cley Cerwyn</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Battle of Winterfell</td>
      <td>20.0</td>
      <td>Theon Greyjoy</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Sack of Winterfell</td>
      <td>618.0</td>
      <td>Ramsay Snow, Theon Greyjoy</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Battle of Oxcross</td>
      <td>6000.0</td>
      <td>Robb Stark, Brynden Tully</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Siege of Storm's End</td>
      <td>5000.0</td>
      <td>Stannis Baratheon, Davos Seaworth</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Battle of the Fords</td>
      <td>20000.0</td>
      <td>Tywin Lannister, Flement Brax, Gregor Clegane,...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sack of Harrenhal</td>
      <td>100.0</td>
      <td>Roose Bolton, Vargo Hoat, Robett Glover</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Battle of the Crag</td>
      <td>6000.0</td>
      <td>Robb Stark, Smalljon Umber, Black Walder Frey</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Battle of the Blackwater</td>
      <td>21000.0</td>
      <td>Stannis Baratheon, Imry Florent, Guyard Morrig...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Battle of Duskendale</td>
      <td>3000.0</td>
      <td>Robertt Glover, Helman Tallhart</td>
    </tr>
    <tr>
      <th>25</th>
      <td>The Red Wedding</td>
      <td>3500.0</td>
      <td>Walder Frey, Roose Bolton, Walder Rivers</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Battle of Castle Black</td>
      <td>100000.0</td>
      <td>Mance Rayder, Tormund Giantsbane, Harma Dogshe...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Retaking of Deepwood Motte</td>
      <td>4500.0</td>
      <td>Stannis Baratheon, Alysane Mormot</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Siege of Dragonstone</td>
      <td>2000.0</td>
      <td>Loras Tyrell, Raxter Redwyne</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Siege of Riverrun</td>
      <td>3000.0</td>
      <td>Daven Lannister, Ryman Fey, Jaime Lannister</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Siege of Raventree</td>
      <td>1500.0</td>
      <td>Jonos Bracken, Jaime Lannister</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Siege of Winterfell</td>
      <td>5000.0</td>
      <td>Stannis Baratheon</td>
    </tr>
  </tbody>
</table>
</div>



<p>To loosely verify we are working with the right data, let's see if our subsets add up</p>


```python
#length of the original should equal that of with and without nulls combined
print 'Actual:', len(battles)
print len(battles)==len(battles_df)+len(battles_w_nulls), len(battles_df), '+', len(battles_w_nulls), '=', len(battles_df)+len(battles_w_nulls)
```

    Actual: 38
    True 24 + 14 = 38
    

<p>Moving on we can now plot the two variables. To do this we add a column to our data that is the count of attacking commanders</p>



```python
#REMINDER: as a learning assignment, figure out how to do this with the appropriate method 
battles_df['attacking_com_count'] = battles_df.apply(lambda row: len(row['attacker_commander'].split(',')), axis=1)

#sort
battles_df = battles_df.sort_values('attacker_size')

x=battles_df['attacker_size'].tolist()
y=battles_df['attacking_com_count'].tolist()

mplt.scatter(x, y)
mplt.show()
```

    C:\Users\Chris\Anaconda2\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    
    


![png](output_24_1.png)

