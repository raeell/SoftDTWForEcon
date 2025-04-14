FROM ubuntu:22.04

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY api/ ./api
COPY data/ ./data
COPY model/ ./model
RUN mkdir plots
CMD ["bash", "-c", "./api/run.sh"]
