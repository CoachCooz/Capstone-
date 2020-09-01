# Los Angeles Dodgers Machine Learning & Data Analysis Project

## Flatiron School

## By - Acusio Bivona

## Abstract

This machine learning & data analysis project utilizes data and statistics from the Los Angeles Dodgers seasons of 2018, 2019, and 2020, to date. The data has been obtained using API's and webscraping. My sources for this data are from www.baseball-reference.com, mlb-data.p.rapidapi.com, and baseballsavant.mlb.com. The types of data obtained include box scores, season hitting stats, season pitching stats, and individual player game logs. This project has been separated into three notebooks - "Data Obtainment & Preprocessing", "Dashboards", and "Modeling". The specifics of each notebook are detailed below in their respective sections.

For this project, there were three primary objectives:
* Use various machine learning models to predict the likelihood of the Dodgers winning or losing their next game based on the box score data of the previous game.
* Create interactive graphs using plotly express to compare specific statistics for a player between their 2018, 2019, and 2020 seasons, if available.
* Use the results of the machine learning models and the visualizations to create simple & direct data-driven recommendations to coaches.

## My Why

Baseball, and many other sports, are becoming more and more data-driven in a variety of ways. Whether it be for game strategy, determining lineups, or projecting a player's performance, data science has entered into a very special marriage with baseball, with the idea of using data to create even the slightest advantage over their opponent. 2020 presents the baseball world with a very interesting scenario as a result of COVID-19. As opposed to the usual and timeless 162 game marathon, the season has been condensed to a 60 game sprint. Stats will be more extreme; pressure will be higher; and there is no taking any series off because every game matters. It is the ultimate test of a team's commitment and the players' focus. So, I asked myself the following question:
>**What if the last two seasons were also only 60 games?**

What could we expect in terms of player performance? Is there any indication that certain players truly get off to "hot starts" in the past, but now that "hot start" is the whole season? Or, we can ask the opposite of that - what if a player is notorious for "cold starts?" What kind of decisions could that force coaches to make? Or is it all just truly random and it really does vary from season to season? These were just some of the questions I asked myself while working on this. But, I know that I'm also not the only person asking this question. So, the most important part of this project, for me, was not just obtaining the results - but to let other Dodgers (or baseball, in general) fans explore and create their own conclusions, themselves. I wanted to provide a simple entry into the world that is data science in baseball.

## Notebook - Data Obtainment & Preprocessing

This notebook is a collection of all the data collected during the 2018 & 2019 seasons, as well as certain data from 2020. 2020 data such as game logs and box scores are not in this notebook because the season is ongoing, and needs to be continuously updated in order to keep the model & interactive dashboards current. As mentioned in the absract, all the data was obtained and preprocessed by me and was collected using API's and webscraping. Once the data was collected, they were saved into pickles so that the information collected can be easily and quickly loaded into other notebooks. There is a folder in this repo called "pickles", which contains all of the pickles that have been saved. The data obtained, in terms of player performance, including pitching stats for the pitchers and hitting stats for the position players. There are no defensive statistics for either group in this project.

## Notebook - Dashboards

This notebook conists of all the game logs for both pitchers and position players from the 2018, 2019, and 2020 seasons, to date. This notebook contains all the code to easily access the interactive graphs that represent specific statistics. There are two viewable statistics for the pitchers and four for the position players, and a function has been built for each one of the following:
> Pitchers:
* ERA (Earned Run Average)
* WHIP (Walks & Hits Allowed Per Inning Pitched)

> Position Players:
* BA (Batting Average)
* OBP (On Base Percentage)
* SLG (Slugging Percentage)
* OPS (On Base Percentage + Slugging Percentage)

The functions are available and need to be ran in order to view the graphs currently. A list of pitcher names and a list of position player names has been provided so that the user knows who's information is available to view. Currently, I am in the process of utilizing Jupyter Dash to create a more user-freindly experience inside the notebook, but that is still a work in progress. Eventually, I would like to host this on the web so that my work can reach an even wider audience.

## Notebook - Modeling

This notebook includes the box score data for all three seasons. I used feature engineering and machine learning to create four models that would try and predict the likelihood of the Dodgers winning or losing their next game, based on the statistics of the previous games. The four models I ran include:
>* Random Forest
* XGBoost
* Naive Bayes
* K-Nearest Neighbors

To date, the results are rather inconclusive and unreliable. For example, the testing accuracy ranges from 44-50%, and the AUC score ranges from .43 to .50. Essentially, flipping a coin and guessing is better than the current performance of these models. I believe the primary reason for this is the very small number of data. There are only 328 current entries, and my algorithm is set to only perform the testing on 66 samples. So, I believe one way to increase performance would be to include more seasons of box scores so that there are more entries. However, that may cause certain features, such as the "Winning Pitcher", to have to be removed due to the fact that, let's say I obtain data from the last 100 seasons, pitchers from years or decades ago won't have any impact on the current results of the games. Like with all machine learning models, improving performance is a trial and error process, and I will continue to try an improve its performance.

In addition to the models, I have created functions to display confusion matrices and ROC-AUC curves for all the models, as well as displaying the most important features for the ensemble methods (Random Forest & XGBoost).

## Recommendations & Future Works

I believe the beauty of this project is that, due to the nature of this season, the recommendations that can be made are very open to interpretation. As a result, my recommendations are moreso ideas rather than specific steps because I believe a wider variety of data analysis is needed to make concrete recommendations. Baseball is a game of patience and situations. For example, how does a batter hit versus a left-handed pitcher versus a right-handed pitcher? Are they better at hitting high fastballs, or low fastballs? For pitchers, does your opponent struggle with off-speed pitches? Do they chase pitches that are high or low, inside or outside? As time goes on, these are areas I would love to explore and really dive into, especially for my Dodgers. But, nonetheless, below are the general recommendations I would make from my work, so far:
>* Observe trends to determine batting orders for hitters
* If someone is really slumping, maybe give them a day off to reset and find their way
* For pitching, use recent performance to influence which relievers will pitch in high-leverage situations
* Observe recent performance of starting pitchers to have a gauge on how many innings they should pitch in their next start.

In addition to exploring more advanced statistics & potentially adding more seasons to create even stronger data-driven recommendations, I would love this project to spread to including more, or all, MLB teams. I believe that would be excellent for all of baseball and its fans, and every one can find something to cheer (and jeer) for, for the team that they support and love. Given ample time and resources. I truly believe that this is just the beginning of something beautiful.


```python

```
