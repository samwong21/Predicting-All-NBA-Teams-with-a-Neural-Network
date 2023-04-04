## Using a Keras Sequential Network to Predict All-NBA Teams


<img width="1000" alt="Screen Shot 2023-04-04 at 3 59 35 PM" src="https://user-images.githubusercontent.com/97067377/229940523-06876c21-6184-4391-8951-5296dd295c32.png">




<ins>**Background**</ins>

Being on the All-NBA Team is an annual award given to the best players in each season by the National Basketball Association. The vote consists of 100 ballots from a global panel of sportswriters and broadcasters. Since 1965, the panel has selected two guards, two forwards, and one center for each of the three teams (All-NBA Team, n.d.). Although the “success” of a player is assumed to be determined on their unbiased game time statistics, because the panel is made of real people, there will always be subjectivity in the ballots casted. When taking into account the approximate 450 players playing in the league each year, appointing the All-NBA Team is easier said than done. Therefore, using a model trained with the previous decades player statistics and who made past All-NBA Teams, our project sought to see if neural networks could be used to determine All-NBA Teams. 

<ins>**Previous Studies**</ins>

Previous studies using machine learning in conjunction with the NBA have primarily focused on predicting tournament brackets and NBA All-Stars. In an article published in Towards Data Science, researchers utilized XGBoost, an implementation of gradient boosted decision trees to classify players into whether or not they were voted to play in the NBA All-Star game. Despite their similarities in name, players are assessed by different metrics when considered for the All-Star game versus the All-NBA Team award. The most distinct of these differences is that starters for the All-Star game are determined by a mix of fan, player, and media voting, while the reserves are chosen by the coaches (Porteous, 2020).  This leads to considerations like social media presence, player popularity among fans, and team record that culminate to selections for the All-Star game straying further away from an unbiased evaluation based solely on player performance. Overall to represent each player, the data scientists decided to use a combination of specific player statistics like points per game, true shooting percentage, and player impact estimate along with external data like team conference rank, team played for, and Avg. Pace (or the average number of possessions a team gets per 48 minutes) (Porteous, 2020). 

<ins>**The Data**</ins>

Our data was scraped from Basketball-Reference.com, a website that has decades worth of NBA statistics.  For the purposes of our project, we decided to use the advanced player stats CSV containing metrics that reflect a more in-depth evaluation of skill and player performance over simple box scores. We combined CSVs from the 2013/2014 season through the 2019/2020 season to serve as training data and seasons 2020/2021 and 2021/2022 for testing data. We created another CSV containing the names of players from the All-NBA teams for all aforementioned seasons. With this, we appended another column to our advanced player stats data frame with each players’ All-NBA team status encoded in binary. A 0 meant a player did not make an All-NBA team and a 1 meant they did. It is important to note that players who played multiple seasons were listed in the data frame for each year they made an appearance in the NBA. Therefore, we also made another column that contained the season year. Using our subjectivity and referencing the Towards Data Science article we determined which metrics from the advanced player stats we would use as a feature vector in our model. We decided on player efficiency rating, true shooting percentage, total rebound percentage, assist percentage, steal percentage, win shares per 48 mins, box plus/minus,  and value over replacement player. After a preliminary look at the data, we noticed that the differences between players who we knew were some of the best in the league and bench players were not always accurately represented in the percentages. Metrics like true shooting percentage, total rebound percentage, assist percentage, and steal percentage could be skewed if an unimportant player got statistics like 1 of 1 rebounds. Therefore, we decided to also pull the average points per game statistic from another CSV and append that to our data as well. All of the feature vector values were normalized before training and the All-NBA team encoding was split into its own dataset.  

<img width="700" alt="Screen Shot 2023-04-03 at 9 30 12 PM" src="https://user-images.githubusercontent.com/97067377/229687361-1534303e-2429-462b-801b-55a484795daa.png" >

***Definitions of all metrics used in feature vector***

<img width="611" alt="Screen Shot 2023-04-03 at 9 33 01 PM" src="https://user-images.githubusercontent.com/97067377/229687713-ae5a8bf1-83cd-4e15-bb2a-1367bff08895.png">




***Snippet of Data (“Player”, “Pos”, and “All-Nba” were dropped in feature vector)***

<ins>**The Model**</ins>

The model was built with the Keras Application Programming Interface (API) that runs on top of TensorFlow. Tensorflow is an end-to-end open source deep learning framework that helps facilitate machine learning applications (Terra, 2023). Together, Keras provides a streamlined interface to build neural networks in TensorFlow. For this project, we used the “keras.Sequential” class to create a sequential model and add our specified layers to it. A sequential neural network has neurons organized in layers that are connected to each other in a feedforward manner. A sequential neural network was ideal for our project because it excels in classification, especially for tasks where the input and output are well-defined and there is a large quantity of training data (Keras Sequential | What Is Keras Sequential? | How to Use?, n.d.). 
	
We built our model with four total layers, including three with the Rectified Linear Unity (ReLU) as an activation function and a dropout rate of 0.2, and an output layer which utilized a sigmoid function to return a value between 0 and 1. The number of units in each layer was 64, 32, 16, and 1. Because our training data was large, but not massive, we decided a relatively simple model with four layers would suffice. We chose the ReLU activation function because of its simplicity, leading to easier computation and faster training. It is also a non-linear activation function that can learn the more complex patterns present in our data (Brownlee, 2019). We thought this would be especially interesting considering the model’s predictions would be compared to human judgment (a panel of judges select All-NBA teams), where metrics are weighted differently depending on subjectivity. The dropout rate of 0.2 helps to prevent overfitting to the training data. In practice, this randomly sets 20% of input values to 0 which adds noise to the data and forces the network to learn more robust features during training. When compiling the model we used a binary cross entropy loss function, which is commonly used in binary classification problems. In layman's terms, binary cross entropy measures the difference between probabilities and the data’s ground truth, therefore as the model improves its predictions the loss decreases and the model becomes better at classifying. Binary cross entropy is optimized using gradient descent, so we used the Adam (Adaptive Moment Estimation) optimizer. The Adam optimizer updates the model's weights during training by adjusting the learning rate of each parameter based on the first and second moments of the gradients (Brownlee, 2017). The lowering of learning rates helps to prevent overshooting. A parameter for validation split was set to 0.2, so our model could evaluate its performance after each epoch during training. In the keras.sequential model, this means that the last 20% of training data is used for validation. It is important to note that the 20% is not chosen at random, but is rather the last 20% of rows in the data (How Does the Validation_split Parameter of Keras' Fit Function Work?, 2018). This was ideal for our project because our data was organized chronologically by season and the ratio of All-NBA team players to non All-NBA team players was extremely low, therefore the 20% split would encapsulate all 15 All-NBA team players from the 2019/2020 season and some. Overall, this type of sequential split would ensure that there would be All-NBA present in the validation set. Using the validation split, the model evaluates its performance on its accuracy and implements gradient descent. The line of code for early stopping allows the model to terminate training five epochs after it notices that validation loss stops improving, otherwise the model will run for 200 epochs. 

Because our model concludes with a sigmoid function layer, the final output will be a number between 0 and 1, which is interpreted as the probability of a player being on an All-NBA team. The higher the prediction number (closer to 1), the more likely a player is on an All-NBA team and vice versa. Therefore, the highest 15 prediction scores (5 players on each of the 3 All-NBA teams) is the model’s guess for who made an All-NBA team. 15 can be adjusted depending on the number of seasons the model is predicting. The mean prediction score for players who actually made an All-NBA team in real life was 0.84 and the mean prediction score for players who did not was 0.03. 

<ins>**Changes to the Model**</ins>

After the first predictions were made, we noticed that many more guards were being selected, while relatively few centers were. After reflecting about what the model is actually doing, we realized that the problem may lay in which metrics the model is weighing more heavily. In real life, different statistics matter more for different positions in basketball. For example, a high total rebound percentage is much more indicative of a good center than it is of a good point guard. This lies simply in the purpose of the position. A center is meant to predominantly play in the key and get rebounds for the team, while a point guard brings the ball up the court. To account for the differences in success metrics, we decided to split our data into guards, forwards, and centers and run three different models so that each would weigh the statistics accordingly. This further reflects how the actual All-NBA teams are picked, two guards, two forwards, and one center for each team. We also noted that Lebron James was not predicted to be on an All-NBA team one of the years he actually was. On the other hand a relatively unimportant player was predicted to be on an All-NBA team. This seemed like a pretty big error, considering James is unanimously one of the best players in the NBA and has comparatively high stats. Recounting back to how the percentages in the feature vector were swayed if a player’s box scores were low, we realized that many players had high stats despite having few game appearances. To counteract this, we cut down our data to only include players that played in 40 or more games. This also reflects current NBA deliberations as the league has been considering a minimum number of games for players to appear in to receive various end of season awards (Wells, n.d.). 

<ins>**Results**</ins>

Note the test set is composed of data from the 2019/2020 and 2020/2021 season. “Ground Truth All-NBA Score” column is whether a player made an All-NBA team in real life (1  = yes, 0 = no). “pred score” column is the prediction score given to the player by the model. “predictions” column is whether or not the player is classified as an All-NBA team player by the model. “correct predictions” column is whether the “predictions” matches the “Ground Truth All-NBA Score”.

<ins>**Guard Model**</ins>

After running the guard model, the final validation loss was 0.0886 and the final validation accuracy was 0.9812 after convergence at 14 epochs. The highest 12 prediction scores were classified as making an All-NBA team. 

<img width="695" alt="Screen Shot 2023-04-03 at 9 33 47 PM" src="https://user-images.githubusercontent.com/97067377/229687815-630f8693-cb83-44d6-be0a-fa25bff066fa.png">

The model predicted **300 out of 306** guards correctly in the test data, meaning it had 98% accuracy. 

<img width="329" alt="Screen Shot 2023-04-03 at 9 34 39 PM" src="https://user-images.githubusercontent.com/97067377/229687922-7d5aba43-a541-44f0-935e-096d6cc88eed.png">

***Image of model predictions across all guards in test data. Blue are correct predictions, each white line is a wrong prediction***

<img width="580" alt="Screen Shot 2023-04-03 at 9 35 07 PM" src="https://user-images.githubusercontent.com/97067377/229687978-666b381f-32d1-4577-a998-26a5035c121b.png">

***Players the guard model predicted incorrectly***

It seemed unusual that 2020/2021 LeBron James had such a low prediction score considering that he is always one of the best players in the NBA. Upon closer inspection, LeBron was put as a point guard in the 2020/2021 season data, when he is usually a forward. This supports our belief that the model weighs statistics differently depending on position and that real All-NBA team voting is at the mercy of judge subjectivity. According to our model, James is not the best guard, but we all know he is one of the best players in the league.

<img width="688" alt="Screen Shot 2023-04-03 at 9 36 18 PM" src="https://user-images.githubusercontent.com/97067377/229688134-5e82a779-b7b6-4bb9-a02c-2be69689a019.png">

***Lebron James’ data for the 2020/2021 Season (note he is listed as PG)***

The guard model predicted **11 out of 13** actual All-NBA team players, meaning it had an 85% accuracy. 
I suspect the extra guard (13 players instead of 12) was because LeBron was listed as a point guard.


<img width="296" alt="Screen Shot 2023-04-03 at 9 38 33 PM" src="https://user-images.githubusercontent.com/97067377/229688388-01675323-e6da-4037-ac28-92f3dd0e6c47.png">

***Image of model predictions for players who actually made the All-NBA team. Blue are correct predictions, each white line is a wrong prediction***


<img width="429" alt="Screen Shot 2023-04-03 at 9 38 54 PM" src="https://user-images.githubusercontent.com/97067377/229688426-954f60cc-f3c2-4e57-a532-40b028f3b7b1.png">

***Players that made an All-NBA team in real life***

To investigate why some players who actually didn’t make an All-NBA team were chosen over players who did actually make an All-NBA team, I looked into the differences between average statistics between our model’s false positives and false negatives. From the images below we can see that the false positives (players who the model predicted to be All-NBA players but weren’t) played, on average, more games, and had a higher average assist percentage than the false negatives (real life All-NBA players the model missed). From this, the inference that games played and assist percentage were weighed more heavily in the guard model.

<img width="756" alt="Screen Shot 2023-04-03 at 9 39 22 PM" src="https://user-images.githubusercontent.com/97067377/229688471-88bb46d3-c4ef-40c5-9edc-e105f82b268f.png">

  


<ins>**Front Court Model**</ins>

After running the front court model, the final validation loss was 0.0764 and the final validation accuracy was 0.965 after convergence at 15 epochs. The highest 12 prediction scores were classified as making an All-NBA team. 
















The model predicted 254 out of 258 front court players correctly in the test data, meaning it had 98% accuracy. 


Image of model predictions for players who actually made the All-NBA team. Green are correct predictions, each white line is a wrong prediction








Players the front court model predicted incorrectly





The guard model predicted 8 out of 10 actual All-NBA team players, meaning it had an 85% accuracy.


Image of model predictions for players who actually made the All-NBA team. Green are correct predictions, each white line is a wrong prediction


 





Players that made an All-NBA team in real life







To investigate why some players who actually didn’t make an All-NBA team were chosen over players who did actually make an All-NBA team, I looked into the differences between average statistics between our model’s false positives and false negatives. From the images below we can see that the false positives (players who the model predicted to be All-NBA players but weren’t) played, on average, more games, and had a higher average assist percentage, usage percentage, and points per game than the false negatives (real life All-NBA players the model missed). From this, the inference that games played, assist percentage, usage percentage, and points per game were weighed more heavily in the front court model.






<ins>**Center Model**</ins>

After running the center model, the final validation loss was 0.1331 and the final validation accuracy was 0.9289 after convergence at 15 epochs. The highest 6 prediction scores were classified as making an All-NBA team. 


















The model predicted 135 out of 136 centers correctly in the test data, meaning it had 99% accuracy. 


Image of model predictions for players who actually made the All-NBA team. Red are correct predictions, each white line is a wrong prediction







Only player the center model predicted incorrectly



The guard model predicted 6 out of 7 actual All-NBA team players, meaning it had an 86% accuracy.


Image of model predictions for players who actually made the All-NBA team. Red are correct predictions, each white line is a wrong prediction







The only wrong prediction the center model made was missing Lebron James who did make an All-NBA team. I believe that this is again because Lebron is listed outside of his usual position of forward in the 2021/2022 season. This would explain why there were 7 centers on the All-NBA team from the test data instead of the expected 6.

There were not any false positives from the center model’s predictions. 

<ins>**Conclusion**</ins>

Some limitations of our model is that it does not take into account a player’s team statistic like their win loss rate. In a real life scenario, if two players had similar stats, the player on the better team would get picked over the player on the worse team. However, this does somewhat reflect the subjectivity of the panel of judges. Team popularity and exposure does impact how judges may view players. Our model is also at the mercy of what the data lists different player positions as. As seen with Lebron James who one season was listed as a point guard, but the next as a center, when he usually is seen as a forward, the model can miss All-NBA players because it places players into a restrictive box based on their listed position. 

Finally, according to our model’s predictions, these are the players who will make the 2023 All-NBA team.  When comparing our results from those of an Bleacher Report article, it is interesting to see that already some of our predictions are in line with what NBA fans are predicting (Favale, n.d.). However, only time will tell. 
Guards


Front Court


Centers

<ins>References</ins>

(n.d.). Basketball-Reference.com: Basketball Statistics & History of Every Team & NBA and WNBA Players. Retrieved March 24, 2023, from https://www.basketball-reference.com/

All-NBA Team. (n.d.). Wikipedia. Retrieved March 23, 2023, from https://en.wikipedia.org/wiki/All-NBA_Team

Brownlee, J. (2017, July 3). Gentle Introduction to the Adam Optimization Algorithm for Deep Learning - MachineLearningMastery.com. Machine Learning Mastery. Retrieved March 24, 2023, from https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

Brownlee, J. (2019, January 9). A Gentle Introduction to the Rectified Linear Unit (ReLU) - MachineLearningMastery.com. Machine Learning Mastery. Retrieved March 24, 2023, from https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

Favale, D. (n.d.). Predicting 2023 All-NBA 1st, 2nd and 3rd Teams. Bleacher Report. Retrieved March 24, 2023, from https://bleacherreport.com/articles/10068690-predicting-2023-all-nba-1st-2nd-and-3rd-teams

How does the validation_split parameter of Keras' fit function work? (2018, September 30). Data Science Stack Exchange. Retrieved March 24, 2023, from https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work

Keras Sequential | What is Keras sequential? | How to use? (n.d.). eduCBA. Retrieved March 24, 2023, from https://www.educba.com/keras-sequential/

Porteous, C. (2020, June 18). Using machine learning to predict NBA All-Stars, Part 1: Data collection. Towards Data Science. Retrieved March 23, 2023, from https://towardsdatascience.com/using-machine-learning-to-predict-nba-all-stars-part-1-data-collection-9fb94d386530

Terra, J. (2023, February 3). Pytorch Vs Tensorflow Vs Keras: Here are the Difference You Should Know. Simplilearn. Retrieved March 24, 2023, from https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article

Wells, A. (n.d.). Report: NBA, NBPA Discuss Rule for Minimum Games Played to Be Eligible for Awards. Bleacher Report. Retrieved March 24, 2023, from https://bleacherreport.com/articles/10068666-report-nba-nbpa-discuss-rule-for-minimum-games-played-to-be-eligible-for-awards
