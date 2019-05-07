# saving some useful commands
docker build -t senti_analyser .
docker run -p 5000:5000 senti_analyser

heroku container:login
heroku create
heroku container:push web --app APP_NAME
heroku container:release web
