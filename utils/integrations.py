from slack_sdk import WebClient

import requests
import smtplib
from email.message import EmailMessage
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def send_slack_alert(webhook_url, message):
    from slack_sdk.webhook import WebhookClient
    client = WebhookClient(webhook_url)
    response = client.send(text=message)
    return response.status_code


def send_email_gmail(to, subject, body, gmail_user, gmail_pass):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.send_message(msg)



def write_to_google_sheet(sheet_name, df):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("configs/google_creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).sheet1
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())



def send_teams_alert(webhook_url, message):
    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload)
    return response.status_code


def create_hubspot_contact(api_key, email, firstname, lastname):
    url = f"https://api.hubapi.com/contacts/v1/contact?hapikey={api_key}"
    data = {
        "properties": [
            {"property": "email", "value": email},
            {"property": "firstname", "value": firstname},
            {"property": "lastname", "value": lastname}
        ]
    }
    return requests.post(url, json=data)
