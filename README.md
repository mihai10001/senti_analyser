# SentimentAnalyser

### Specifications  
Web application that analyses sentiment from any given input string (any text) and gives different insights based on the chosen method  
Has multiple methods, that can contain both pre-trained models or a self trained model by me (not that great accuracy)  


### Implementation  
SPA application made with Python, using the Flask framework in order to build the backend and serve static content (HTML pages)  
Uses known Python libraries for Natural Language Processing and sentiment analysis and detection like TextBlob, NLTK or NLTK Vader  
Contains a self trained model based on data from a movie review data set  

### How to run

**Application**
1. cd /app
2. pip3 install -r requirements.txt
3. python3 main.py


**Docker/Docker-compose**
Just build and run the docker image


**Link**
For easier access, follow this link: [Press me](https://senti-analyser.herokuapp.com/) 


### Todo: 
UI more user friendly
