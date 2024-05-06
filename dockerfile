# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt or directly
RUN pip install --no-cache-dir pinecone-client==2.2.4 pinecone-text==0.8.0 openai==1.9.0 llama-index==0.9.34 notion-client==2.2.1 fastapi==0.100.0 msal==1.22.0

# Make port 80 available to the world outside this container
# (Optional: If your app uses a port, you can expose it with this command)
# EXPOSE 80

# Define environment variable
# (Optional: Use this section to set any environment variables you need)
# ENV NAME World


# Run upsertNotionSparseDense.py when the container launches
CMD ["python", "./upsertNotionSparseDense.py"]
