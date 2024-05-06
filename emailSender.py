import requests
import json
from auth import acquire_access_token_without_user

def emailSender(
        access_token: str,
        email_address: str,
        ccRecipients: list[str],
        my_email: str,
        question: str,
        answer: str,
        references: str,
        rating: str
    ):
    headers = {'Authorization': 'Bearer ' + access_token,
               'Content-Type' : 'application/json'}
    
    log = ""
    try:
        with open('./upsertNotion.log', 'r') as file:
            log = file.read()
    except FileNotFoundError:
        log = "The file was not found."
    except Exception as e:
        log = f"An error occurred: {e}"
    
    msg = {
        "message": {
            "subject": "ChatDNAS Knowledge Upsert Log",
            "body": {
                "contentType": "Text",
                "content": log
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "address": email_address
                    }
                }
            ],
            "ccRecipients": [{"emailAddress": {"address": recipients}} for recipients in ccRecipients]
        },
        "saveToSentItems": "true"
    }
    
    status = requests.post(f'https://graph.microsoft.com/v1.0/users/{my_email}/sendMail', 
                           headers=headers, json=json.loads(json.dumps(msg)))
    return status

def sendEmail(
        question: str,
        answer: str,
        references: str,
        rating: str):
    ccRecipients = [
        "lguo@wearetheone.com"
    ]
    access_token = acquire_access_token_without_user()

    status = emailSender(
        access_token,
        "it@wearetheone.com",
        ccRecipients,
        "it@wearetheone.com",
        question,
        answer,
        references,
        rating
    )
    if status.status_code == 202:
        return "Email sent successfully."
    else:
        return status