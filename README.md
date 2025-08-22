# Top-G

■ The overall problem being solved:

The main issue in hospitals is their lack of clarity in telling the customers how long the stay is going to be. This project uses xgboost a machine learning model to take many features in to tell us how long the stay will be and we also have administrative delay which is the delay that is caused due to complications in the hospital billing system or insurance issues so this project tells us the total no of days factoring all this in.

■ The steps taken to solve the problem.

We used xgboost model , we trained it with the given dataset then we input the features from the user and make prediction on how long the stay will be. We then considered the features along with the stay to find the estimated cost, using the estimated cost along with a score system we found the administrative delay and then added it to the stay to find the total days and cost respectively.

■ The libraries and frameworks used:

The libraries used are:
1.pandas
2.sklearn
3.xgboost
4.seaborn
5.matplotlib

The frameworks we used is streamlit.

■ How to run the code:

The jupyter notebook code can be run in either jupyter or colab to train the model and predict the values.
The streamlit can be run via vs code after putting the code in the platform and it gives the models predictions on the stay duration , adminstrative delay , estimated costs etc in the form of a dashboard.
