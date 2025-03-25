from flask import Flask, render_template, request, flash, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import random
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import time
import pandas as pd

app = Flask(__name__)
app.secret_key = 'ksidkdsqhriivvkr'

DATABASE = 'users.db'
EXCEL_FILE = 'users.xlsx'
load_dotenv()

# Email configuration
sender_email = "techarmycustomercare@gmail.com"
sender_password = "ywmuazkbexdukndg"

# Database connection
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        return None

# Initialize the database
def init_db():
    conn = get_db_connection()
    if conn is None:
        print("Failed to initialize database due to connection error.")
        return
    
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            mobile TEXT NOT NULL,
            password TEXT NOT NULL,
            otp TEXT,
            otp_expiry INTEGER
        )
    ''')
    
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'otp' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN otp TEXT')
    if 'otp_expiry' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN otp_expiry INTEGER')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Save user data to Excel
def save_to_excel(email, mobile, password):
    user_data = {
        'Email': [email],
        'Mobile': [mobile],
        'Password': [password],
        'Registration Time': [time.strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    df = pd.DataFrame(user_data)
    
    if os.path.exists(EXCEL_FILE):
        existing_df = pd.read_excel(EXCEL_FILE)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(EXCEL_FILE, index=False)
    else:
        df.to_excel(EXCEL_FILE, index=False)
    
    print(f"User data saved to {EXCEL_FILE}: {email}")

# Generate a 6-digit OTP
def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

# Send OTP via email with improved error handling
def send_otp_email(email, otp, max_retries=3):
    subject = "Password Reset OTP"
    body = f"Your OTP for password reset is: {otp}\n\nPlease use this OTP to reset your password. It is valid for 10 minutes."
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    if not sender_password:
        print("Error: EMAIL_PASSWORD is not set")
        return False

    for attempt in range(max_retries):
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"OTP {otp} sent successfully to {email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"Attempt {attempt + 1} failed: Authentication Error - {e}")
            return False
        except smtplib.SMTPException as e:
            print(f"Attempt {attempt + 1} failed: SMTP Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Unexpected Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False

# Send registration success email
def send_registration_email(email, max_retries=3):
    subject = "Registration Successful"
    body = "Dear User,\n\nCongratulations! Your registration with Patient Care Analysis Outreach is successful.\n\nYou can now log in using your email and password to access our services.\n\nThank you,\nTecharmy Team"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    if not sender_password:
        print("Error: EMAIL_PASSWORD is not set")
        return False

    for attempt in range(max_retries):
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"Registration email sent successfully to {email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"Attempt {attempt + 1} failed: Authentication Error - {e}")
            return False
        except smtplib.SMTPException as e:
            print(f"Attempt {attempt + 1} failed: SMTP Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Unexpected Error - {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False

# Symptom-to-output mapping
def get_health_outputs(symptoms):
    symptom_map = {
        "fever": {
            "disease": "Common Cold or Flu",
            "medicines": "Paracetamol, Ibuprofen",
            "diet_plan": "Stay hydrated, consume warm fluids like soup, avoid cold foods"
        },
        "stomach pain": {
            "disease": "Gastritis or IBS",
            "medicines": "Antacids, Buscopan",
            "diet_plan": "Eat bland foods like rice and bananas, avoid spicy or fatty foods"
        },
        "headache": {
            "disease": "Tension Headache or Migraine",
            "medicines": "Aspirin, Sumatriptan",
            "diet_plan": "Stay hydrated, limit caffeine, eat magnesium-rich foods like nuts"
        }
    }
    symptoms_lower = symptoms.lower()
    for key in symptom_map:
        if key in symptoms_lower:
            return symptom_map[key]
    return {
        "disease": "Unknown (consult a doctor)",
        "medicines": "Consult a healthcare provider",
        "diet_plan": "Maintain a balanced diet and consult a professional"
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email.', 'error')
            return render_template('reset_password.html')

        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('reset_password.html')

        try:
            cursor = conn.cursor()
            user = cursor.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if not user:
                flash('Email not found in our records. Please register first.', 'error')
                return render_template('reset_password.html')

            otp = generate_otp()
            expiry_time = int(time.time()) + 600  # 10 minutes
            cursor.execute('UPDATE users SET otp = ?, otp_expiry = ? WHERE email = ?', (otp, expiry_time, email))
            conn.commit()

            if send_otp_email(email, otp):
                flash('OTP sent to your email. Please check your inbox.', 'success')
                return redirect(url_for('verify_otp', email=email))
            else:
                flash('Failed to send OTP. Check email settings or try again later.', 'error')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
        finally:
            conn.close()
        
        return render_template('reset_password.html')
    return render_template('reset_password.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if not email:
        flash('Invalid request.', 'error')
        return redirect(url_for('reset_password'))
    
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        verify_password = request.form.get('verify_password')

        if not all([entered_otp, new_password, verify_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('verify_otp.html', email=email)
        if new_password != verify_password:
            flash('Passwords do not match.', 'error')
            return render_template('verify_otp.html', email=email)

        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('verify_otp.html', email=email)
        
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if not user:
            flash('User not found. Please register first.', 'error')
            conn.close()
            return render_template('verify_otp.html', email=email)
        
        current_time = int(time.time())
        if user['otp'] is None or user['otp_expiry'] is None:
            flash('No OTP generated for this email. Please request a new one.', 'error')
        elif user['otp'] == entered_otp and current_time < user['otp_expiry']:
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            conn.execute('UPDATE users SET password = ?, otp = NULL, otp_expiry = NULL WHERE email = ?', 
                        (hashed_password, email))
            conn.commit()
            flash('Password reset successfully! Please log in.', 'success')
            conn.close()
            return redirect(url_for('home'))
        elif current_time >= user['otp_expiry']:
            flash('OTP has expired. Please request a new one.', 'error')
        else:
            flash('Invalid OTP. Please try again.', 'error')
        
        conn.close()
        return render_template('verify_otp.html', email=email)
    return render_template('verify_otp.html', email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        verify_password = request.form.get('verify_password')
        if not all([email, mobile, password, verify_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        if password != verify_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('register.html')
        
        try:
            conn.execute('INSERT INTO users (email, mobile, password) VALUES (?, ?, ?)',
                         (email, mobile, hashed_password))
            conn.commit()
            save_to_excel(email, mobile, hashed_password)
            
            # Send registration success email
            if send_registration_email(email):
                flash('Registration successful! A confirmation email has been sent to your inbox.', 'success')
            else:
                flash('Registration successful, but failed to send confirmation email. Please contact support.', 'success')
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.', 'error')
        finally:
            conn.close()
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html', show_search=False)
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('login.html', show_search=False)
        
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_email'] = email
            flash('Login successful!', 'success')
            return render_template('login.html', show_search=True)
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html', show_search=False)
    return render_template('login.html', show_search=False)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if not session.get('logged_in'):
        flash('Please log in to use the search feature.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms:
            flash('Please enter symptoms to search.', 'error')
            return render_template('login.html', show_search=True)
        
        outputs = get_health_outputs(symptoms)
        return render_template('login.html', show_search=True, symptoms=symptoms, outputs=outputs)
    return render_template('login.html', show_search=True)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True)





new code starts here:
from flask import Flask, render_template, request, flash, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import random
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import time
import pandas as pd

app = Flask(__name__)
app.secret_key = 'ksidkdsqhriivvkr'

DATABASE = 'users.db'
EXCEL_FILE = 'users.xlsx'
load_dotenv()

# Email configuration
sender_email = "techarmycustomercare@gmail.com"
# It's recommended to store your email password in an environment variable
sender_password = os.getenv('EMAIL_PASSWORD', 'your_default_password_here')

# Database connection
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        return None

# Initialize the database
def init_db():
    conn = get_db_connection()
    if conn is None:
        print("Failed to initialize database due to connection error.")
        return

    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            mobile TEXT NOT NULL,
            password TEXT NOT NULL,
            otp TEXT,
            otp_expiry INTEGER
        )
    ''')

    # Ensure 'otp' and 'otp_expiry' columns exist
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'otp' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN otp TEXT')
    if 'otp_expiry' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN otp_expiry INTEGER')

    conn.commit()
    conn.close()
    print("Database initialized successfully!")

# Save user data to Excel
def save_to_excel(email, mobile, password):
    user_data = {
        'Email': [email],
        'Mobile': [mobile],
        'Password': [password],
        'Registration Time': [time.strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    df = pd.DataFrame(user_data)
    
    if os.path.exists(EXCEL_FILE):
        existing_df = pd.read_excel(EXCEL_FILE)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(EXCEL_FILE, index=False)
    else:
        df.to_excel(EXCEL_FILE, index=False)
    
    print(f"User data saved to {EXCEL_FILE}: {email}")

# Generate a 6-digit OTP
def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

# Send email (used for OTP, registration, and health outputs)
def send_email(subject, body, recipient_email):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)  # Use the correct `msg` variable
        print(f"Email sent successfully to {recipient_email}")
        return True  # Properly aligned within the function
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False  # Properly aligned within the function


# Symptom-to-output mapping
def get_health_outputs(symptoms):
    symptom_map = {
    "fever": {
        "disease": "Common Cold or Flu",
        "medicines": "Paracetamol, Ibuprofen",
        "diet_plan": "Stay hydrated, consume warm fluids like soup, avoid cold foods"
    },
    "stomach pain": {
        "disease": "Gastritis or IBS",
        "medicines": "Antacids, Buscopan",
        "diet_plan": "Eat bland foods like rice and bananas, avoid spicy or fatty foods"
    },
    "headache": {
        "disease": "Tension Headache or Migraine",
        "medicines": "Aspirin, Sumatriptan",
        "diet_plan": "Stay hydrated, limit caffeine, eat magnesium-rich foods like nuts"
    },
    "cough": {
        "disease": "Bronchitis or Respiratory Infection",
        "medicines": "Cough syrup, Mucolytics",
        "diet_plan": "Drink warm fluids, avoid cold or sugary foods, and use a humidifier"
    },
    "fatigue": {
        "disease": "Anemia or Chronic Fatigue Syndrome",
        "medicines": "Iron supplements, Vitamin B12",
        "diet_plan": "Eat iron-rich foods (spinach, red meat), and stay hydrated"
    },
    "chest pain": {
        "disease": "Heart Disease or GERD",
        "medicines": "Aspirin, Nitroglycerin (if cardiac), Antacids (if GERD)",
        "diet_plan": "Avoid heavy meals, limit sodium, and stay hydrated"
    },
    "back pain": {
        "disease": "Muscle Strain or Herniated Disc",
        "medicines": "Ibuprofen, Muscle relaxants",
        "diet_plan": "Apply hot/cold packs, perform gentle stretching, and maintain good posture"
    },
    "nausea": {
        "disease": "Food Poisoning or Motion Sickness",
        "medicines": "Domperidone, Ondansetron",
        "diet_plan": "Eat small, bland meals, avoid oily or spicy foods"
    },
    "joint pain": {
        "disease": "Arthritis or Gout",
        "medicines": "NSAIDs, Colchicine",
        "diet_plan": "Consume anti-inflammatory foods like turmeric and ginger, avoid processed foods"
    },
    "sore throat": {
        "disease": "Tonsillitis or Viral Infection",
        "medicines": "Lozenges, Ibuprofen",
        "diet_plan": "Gargle with warm salt water, drink herbal teas, and avoid cold drinks"
    },
    "shortness of breath": {
        "disease": "Asthma or COPD",
        "medicines": "Bronchodilators, Steroids",
        "diet_plan": "Avoid allergens, stay hydrated, and use a humidifier"
    },
    "dizziness": {
        "disease": "Vertigo or Low Blood Pressure",
        "medicines": "Meclizine, Fludrocortisone",
        "diet_plan": "Increase salt intake (if BP-related), stay hydrated, avoid sudden movements"
    },
    "constipation": {
        "disease": "Irritable Bowel Syndrome (IBS) or Dehydration",
        "medicines": "Laxatives, Fiber supplements",
        "diet_plan": "Eat fiber-rich foods, drink plenty of water, and exercise regularly"
    },
    "diarrhea": {
        "disease": "Gastroenteritis or Food Poisoning",
        "medicines": "Loperamide, Oral Rehydration Solution (ORS)",
        "diet_plan": "Increase fluid intake, avoid dairy and spicy foods, eat bananas and rice"
    },
    "fever, cough": {
        "disease": "Common Flu",
        "medicines": "Paracetamol, Cough Syrup",
        "diet_plan": "Stay hydrated, drink warm fluids, and rest adequately"
    },
    "fever, rash": {
        "disease": "Measles or Chickenpox",
        "medicines": "Antihistamines, Paracetamol",
        "diet_plan": "Stay hydrated, consume soft foods, and avoid scratching the rash"
    },
    "stomach pain, diarrhea": {
        "disease": "Food Poisoning or Gastroenteritis",
        "medicines": "Loperamide, ORS",
        "diet_plan": "Eat bland foods like rice and bananas, avoid dairy and spicy foods"
    },
    "headache, dizziness": {
        "disease": "Migraine or Vertigo",
        "medicines": "Aspirin, Meclizine",
        "diet_plan": "Stay hydrated, limit caffeine, and eat magnesium-rich foods"
    }
}

    symptoms_lower = symptoms.lower()
    for key in symptom_map:
        if key in symptoms_lower:
            return symptom_map[key]
    return {
        "disease": "Unknown (consult a doctor)",
        "medicines": "Consult a healthcare provider",
        "diet_plan": "Maintain a balanced diet and consult a professional"
    }

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            flash('Please enter your email.', 'error')
            return render_template('reset_password.html')

        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('reset_password.html')

        try:
            cursor = conn.cursor()
            user = cursor.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if not user:
                flash('Email not found in our records. Please register first.', 'error')
                return render_template('reset_password.html')

            otp = generate_otp()
            expiry_time = int(time.time()) + 600  # 10 minutes
            cursor.execute('UPDATE users SET otp = ?, otp_expiry = ? WHERE email = ?', (otp, expiry_time, email))
            conn.commit()

            if send_email("Password Reset OTP", f"Your OTP is: {otp}\nIt is valid for 10 minutes.", email):
                flash('OTP sent to your email. Please check your inbox.', 'success')
                return redirect(url_for('verify_otp', email=email))
            else:
                flash('Failed to send OTP. Try again later.', 'error')
        finally:
            conn.close()
        
        return render_template('reset_password.html')
    return render_template('reset_password.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if not email:
        flash('Invalid request.', 'error')
        return redirect(url_for('reset_password'))
    
    if request.method == 'POST':
        entered_otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        verify_password = request.form.get('verify_password')

        if not all([entered_otp, new_password, verify_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('verify_otp.html', email=email)
        if new_password != verify_password:
            flash('Passwords do not match.', 'error')
            return render_template('verify_otp.html', email=email)

        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('verify_otp.html', email=email)
        
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if not user:
            flash('User not found. Please register first.', 'error')
            conn.close()
            return render_template('verify_otp.html', email=email)
        
        current_time = int(time.time())
        if user['otp'] == entered_otp and current_time < user['otp_expiry']:
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            conn.execute('UPDATE users SET password = ?, otp = NULL, otp_expiry = NULL WHERE email = ?', 
                        (hashed_password, email))
            conn.commit()
            flash('Password reset successfully! Please log in.', 'success')
            conn.close()
            return redirect(url_for('home'))
        else:
            flash('Invalid or expired OTP.', 'error')
        
        conn.close()
        return render_template('verify_otp.html', email=email)
    return render_template('verify_otp.html', email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        password = request.form.get('password')
        verify_password = request.form.get('verify_password')
        if not all([email, mobile, password, verify_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        if password != verify_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('register.html')
        
        try:
            conn.execute('INSERT INTO users (email, mobile, password) VALUES (?, ?, ?)',
                         (email, mobile, hashed_password))
            conn.commit()
            save_to_excel(email, mobile, hashed_password)
            
            if send_email("Registration Successful", "Welcome to our platform!", email):
                flash('Registration successful! A confirmation email has been sent to your inbox.', 'success')
            else:
                flash('Registration successful, but failed to send confirmation email.', 'warning')
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.', 'error')
        finally:
            conn.close()
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash('Please fill in all fields.', 'error')
            return render_template('login.html', show_search=False)
        conn = get_db_connection()
        if conn is None:
            flash('Database connection failed.', 'error')
            return render_template('login.html', show_search=False)
        
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['user_email'] = email
            flash('Login successful!', 'success')
            return render_template('login.html', show_search=True)
        else:
            flash('Invalid email or password.', 'error')
            return render_template('login.html', show_search=False)
    return render_template('login.html', show_search=False)

@app.route('/search', methods=['GET', 'POST'])
def search():
    if not session.get('logged_in'):
        flash('Please log in to use the search feature.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms:
            flash('Please enter symptoms to search.', 'error')
            return render_template('login.html', show_search=True)
        
        outputs = get_health_outputs(symptoms)
        user_email = session.get('user_email')
        
        # Send email with health outputs
        if user_email:
            if send_email("Your Health Analysis Results", 
                          f"Dear User,\n\nYou recently searched for symptoms: {symptoms}\n\n"
                          f"Disease: {outputs['disease']}\n"
                          f"Recommended Medicines: {outputs['medicines']}\n"
                          f"Diet Plan: {outputs['diet_plan']}\n\n"
                          "Note: Please follow our instructions for quick recovery.use our portal to connect us.\n\n"
                          "Thank you,\nTecharmy Team", user_email):
                flash('Analysis complete! Results have been sent to your email.', 'success')
            else:
                flash('Analysis complete, but failed to send results to email.', 'warning')
        
        return render_template('login.html', show_search=True, symptoms=symptoms, outputs=outputs)
    return render_template('login.html', show_search=True)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

# Initialize database on startup
init_db()

if __name__ == '__main__':
    app.run(debug=True)
