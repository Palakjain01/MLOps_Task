
import smtplib, ssl

port = 465 #For SSL
smtp_server ="smtp.gmail.com"
sender_email="iampalakjain01@gmail.com"    #Sender's Mail Address
receiver_email="itspalak19@gmail.com"      #Receiver's Mail Address
password="xecbeupbulzfwpos"
if accuracy > 90:
    message="""    Subject: Report | Prediction Program
    
    CONGRATULATIONS! 
    Your code achieved{}% accuracy.""".format(accuracy)
else:
    message="""    Subject: Report | Prediction Program
    
    Train Again!
    Your code got {}% accuracy.""".format(accuracy)
    
context=ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email,password)
    server.sendmail(sender_email, receiver_email, message)
    
    
