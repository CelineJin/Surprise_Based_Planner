FROM python:3.8

RUN apt-get update -y
RUN apt-get install -y vim
RUN apt-get install -y gcc g++ gfortran subversion patch wget git make

# Install python packages
WORKDIR /plan/
RUN pip install numpy
RUN pip install json5
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install scipy
RUN pip install pyDOE
RUN pip install Flask
RUN pip install requests

# Copy files
COPY ./* /plan/

EXPOSE 80
ENTRYPOINT ["python3", "app.py"]
