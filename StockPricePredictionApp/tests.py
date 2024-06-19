from django.test import TestCase

import smtplib
from email.mime.text import MIMEText

def send_verification_email(to_email, verification_link):
    # Set up the SMTP server
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    file='C:/Users/arvin/OneDrive/Desktop/Project/Django Project/Stock_Price_Predicton-main/StockPricePrediction/pass.txt'
    with open(file,'r') as f:
        paco=f.read()
    password=str(paco)
    smtp_username = 'vp6792338@gmail.com'
    smtp_password = password

    # Set up the email content
    subject = 'Email Verification'
    body = f'Click the following link to verify your email: {verification_link}'
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = smtp_username
    message['To'] = to_email

    # Connect to the SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        refused_recipients =server.sendmail(smtp_username, [to_email], message.as_string())
        if not refused_recipients:
            print(f"Email sent successfully to {to_email}")
        else:
            print(f"Failed to send email to {to_email}. Refused recipients: {refused_recipients}")
# Example usage
verification_link = ' https://t6x515k7-8000.inc1.devtunnels.ms/'
email_to=input('enter email')
send_verification_email(email_to, verification_link)

