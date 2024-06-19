import random
from email.message import EmailMessage
from smtplib import SMTP_SSL


def mail_task(email):
        host='smtp.gmail.com'
        port=465
        user_sender='vp6792338@gmail.com'
        user_receiver=email
        mail_msg='Send email Succssesfully'
        file='C:/Users/Kaustubh/OneDrive/Desktop/StockPricePrediction2/pass.txt'
        with open(file,'r') as f:
           paco=f.read()
        password=str(paco)
        #print(password)
        vary_code=''.join(str(random.randint(0,9)) for i in range(0,6))
        print('vary_code: ',vary_code)
        em=EmailMessage()
        em['From']=user_sender
        em['To']=user_receiver
        em['Subject']="vary_code"
        em.set_content('click here to page https://t6x515k7-8000.inc1.devtunnels.ms/')
        try:
            with SMTP_SSL(host,port) as stmp:
                stmp.login(user_sender,password)
                send_conf= stmp.sendmail(user_sender,user_receiver,em.as_string())
            if send_conf.__sizeof__: 
                print(send_conf)
                return mail_msg
            else:
                error_msg='Not send email successfully'
                return error_msg
        except Exception as e:
            return f'Error: {str(e)}'