import smtplib
from email.mime.text import MIMEText

sender_email = "techarmycustomercare@gmail.com"
sender_password = "your_app_password_here"  # Replace with your App Password
recipient_email = "test@example.com"

msg = MIMEText("This is a test email.")
msg['Subject'] = "Test Email"
msg['From'] = sender_email
msg['To'] = recipient_email

try:
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        print("Email sent successfully.")
except Exception as e:
    print(f"Failed to send email: {e}")
