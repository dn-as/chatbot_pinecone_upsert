## Chatbot Pinecone Upsert
This repo contains code upsert vectors into Pincone DB.
It uses text-embedding-ada-002 to convert documents to vector embeddings.
### To execute
- Build Docker image
  - docker build -t notion-upsert . 
- Install AWS CLI
- Log into AWS CLI
- Push Docker image to AWS ECR
  - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 536666892120.dkr.ecr.us-east-1.amazonaws.com/
    - 536666892120 is the account we are using.
    - us-east-1 is the region of AWS
- Tag the image
  - docker tag notion-upsert:latest 536666892120.dkr.ecr.us-east-1.amazonaws.com/notion-update:latest
- Push image to AWS
  - docker push 536666892120.dkr.ecr.us-east-1.amazonaws.com/notion-update:latest
- Use AWS ECS to run this image
- Can also set up Scheduled Task
  - cron(0 3 ? * SAT *)
    - Runs every Saturday at 3AM UTC (10PM / 11PM EDT (New York City))