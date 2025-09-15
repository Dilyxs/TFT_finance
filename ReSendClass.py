import os
import resend
from dotenv import load_dotenv

class EmailSender:
    def __init__(self, ApiKey=None, prefix=None):
        load_dotenv()
        self.API_KEY = ApiKey if ApiKey else os.getenv("RESEND_API")
        if not self.API_KEY:
            raise ValueError("no key found")

        resend.api_key = self.API_KEY
        prefix = prefix if prefix else "general"

        self.From = f"{prefix}@notifications.adsayan.com"
        self.DefaultTo = "adsayan206@gmail.com"

    def SendEmail(self, subject, message, ToWhom=None, attachmentFileName = None):
        To = self.DefaultTo if not ToWhom else ToWhom
        params = {
            "from": self.From,   
            "to": [To],
            "subject": subject,
            "html": f"<strong>{message}</strong>",
            "attachments":[] if not attachmentFileName else [{"filename":attachmentFileName, "content":open(attachmentFileName, "rb").read()}]
        }

        try:
            email = resend.Emails.send(params)
            print("email sent")
            return email
        except Exception as e:
            print(f"failed to send email as{e}")
            raise 

