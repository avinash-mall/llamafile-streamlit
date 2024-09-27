# Step 1: Use an official Python runtime as a parent image
FROM python:3.12-slim

# Step 2: Clone git repo and update
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends git-all
RUN git clone https://github.com/avinash-mall/llamafile-streamlit.git
RUN cd llamafile-streamlit

# Step 3: Set the working directory in the container
WORKDIR /llamafile-streamlit

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Download Models
RUN python3 downloads.py

# Step 6: Expose the port that the Streamlit app will run on
EXPOSE 8501

# Step 7: Define the command to run the app using streamlit
CMD ["streamlit", "run", "app.py"]
