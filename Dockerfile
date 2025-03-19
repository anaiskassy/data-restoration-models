FROM python:3.10.6-buster

COPY data_restoration data_restoration
COPY models models
COPY api api
COPY requirements.txt requirements.txt

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", $PORT]
#EXPOSE 8080
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
#The image should contain:
#YESthe same Python version of your virtual env
#YESall the directories from the /taxifare project needed to run the API
#YESthe list of dependencies (don’t forget to install them!)

#The web server should:
#launch when a container is started from the image
#listen to the HTTP requests coming from outside the container (see host parameter)
#be able to listen to a specific port defined by an environment variable $PORT (see port parameter)
