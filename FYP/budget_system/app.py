import os
import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
import webbrowser


from nlp_utils import analyze_message
from predict import predict_all_horizon          # SIMPLE
from predict_com import predict_all_horizons_multi     # ADVANCED


# =========================
# ENV SETUP
# =========================
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

from functools import wraps

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


ADMIN_EMAILS = {
    "admin1@example.com",
    "admin2@example.com",
    "admin3@example.com"
}

@app.route("/admin")
@admin_required
def admin_dashboard():
    return render_template("admin_dashboard.html")

@app.route("/admin/users")
@admin_required
def admin_users():
    res = supabase.table("users").select("*").execute()
    users = res.data or []

    users = [u for u in users if u["email"] not in ADMIN_EMAILS]

    return render_template("admin_users.html", users=users)




@app.route("/admin/users/toggle/<int:user_id>", methods=["POST"])
@admin_required
def toggle_user(user_id):
    # prevent admin self-disable
    admin_email = session.get("email")

    user = (
        supabase.table("users")
        .select("is_active, email")
        .eq("id", user_id)
        .single()
        .execute()
        .data
    )

    if user and user["email"] not in ADMIN_EMAILS:
        supabase.table("users").update({
            "is_active": not user["is_active"]
        }).eq("id", user_id).execute()

    return redirect(url_for("admin_users"))


@app.route("/admin/users/edit/<int:user_id>", methods=["GET", "POST"])
@admin_required
def edit_user(user_id):
    res = (
        supabase.table("users")
        .select("*")
        .eq("id", user_id)
        .single()
        .execute()
    )

    user = res.data
    if not user:
        return "User not found", 404

    if request.method == "POST":
        supabase.table("users").update({
            "name": request.form["name"].strip(),
            "email": request.form["email"].strip()
        }).eq("id", user_id).execute()

        return redirect(url_for("admin_users"))

    return render_template("admin_edit_user.html", user=user)

@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    supabase.table("users").delete().eq("id", user_id).execute()
    return redirect(url_for("admin_users"))

# =========================
# DATE FILTER
# =========================
def format_datetime(value, format_string='%Y-%m-%d'):
    if not value:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.strftime(format_string)
    except Exception:
        return str(value).split('T')[0]

app.jinja_env.filters['datetimeformat'] = format_datetime


# =========================
# AUTH ROUTES
# =========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        try:
            auth = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            # ✅ Copy to public.users
            supabase.table("users").insert({
                "auth_id": auth.user.id,
                "email": email
            }).execute()

            return redirect(url_for('login'))

        except Exception as e:
            return render_template('register.html', error=str(e))

    return render_template('register.html')



@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/home_admin')
def home_admin():
    return render_template('home_admin.html')

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        try:
            auth = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            # 🔍 Check public.users status
            user = (
                supabase.table("users")
                .select("id, is_active")
                .eq("auth_id", auth.user.id)
                .single()
                .execute()
                .data
            )

            if not user or not user["is_active"]:
                return render_template(
                    "login.html",
                    error="Your account has been deactivated by admin."
                )

            session.clear()
            session["user"] = auth.user.id
            session["email"] = email

            # Admin detection
            if email in ADMIN_EMAILS:
                session["is_admin"] = True
                return redirect(url_for("home_admin"))

            return redirect(url_for("home"))

        except Exception:
            return render_template("login.html", error="Invalid login")

    return render_template("login.html")






@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# =========================
# DASHBOARD (SIMPLE)
# =========================
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    if request.method == 'POST':
        message = request.form.get('message', '')
        parsed = analyze_message(message) or []

        for t in parsed:
            if t.get('amount') and t.get('type') != 'unknown':
                supabase.table('transactions').insert({
                    "user_id": user_id,
                    "description": message,
                    "type": t['type'],
                    "amount": t['amount'],
                    "category": t.get('category')
                }).execute()

        return redirect(url_for('dashboard'))

    res = supabase.table('transactions').select('*').eq('user_id', user_id).execute()
    transactions = res.data or []

    df = pd.DataFrame(transactions)
    if not df.empty:
        df.rename(columns={'created_at': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df.dropna(subset=['date', 'amount'], inplace=True)
        df.sort_values(by='date', inplace=True)

    prediction_message, next_day, next_week, next_month, metrics = predict_all_horizon(df)

    total_expense = sum(t['amount'] for t in transactions if t['type'] == 'expense')
    total_income = sum(t['amount'] for t in transactions if t['type'] == 'income')

    return render_template(
        'dashboard.html',
        transactions=transactions,
        total_expense=total_expense,
        total_income=total_income,
        prediction_message=prediction_message,
        next_day=f"{next_day:,.2f}",
        next_week=f"{next_week:,.2f}",
        next_month=f"{next_month:,.2f}",
        metrics=metrics
    )


# =========================
# DASHBOARD (ADVANCED)
# =========================
@app.route('/dashboard1', methods=['GET', 'POST'])
def dashboard1():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    res = supabase.table('transactions').select('*').eq('user_id', user_id).execute()
    transactions = res.data or []

    df = pd.DataFrame(transactions)
    if not df.empty:
        df.rename(columns={'created_at': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.normalize()
        df.dropna(subset=['date', 'amount'], inplace=True)
        df.sort_values(by='date', inplace=True)

    prediction_message, next_day, next_week, next_month, model_results = (
        predict_all_horizons_multi(df)
    )

    total_expense = sum(t['amount'] for t in transactions if t['type'] == 'expense')
    total_income = sum(t['amount'] for t in transactions if t['type'] == 'income')

    return render_template(
        'dashboard1.html',
        transactions=transactions,
        total_expense=total_expense,
        total_income=total_income,
        prediction_message=prediction_message,
        next_day=f"{next_day:,.2f}",
        next_week=f"{next_week:,.2f}",
        next_month=f"{next_month:,.2f}",
        model_results=model_results
    )


# =========================
# DELETE
# =========================
@app.route('/delete/<int:transaction_id>', methods=['POST'])
def delete_transaction(transaction_id):
    user_id = session.get('user')
    supabase.table('transactions').delete().eq('id', transaction_id).eq('user_id', user_id).execute()
    return redirect(url_for('home'))


@app.route("/edit/<int:transaction_id>", methods=["GET", "POST"])
def edit_transaction(transaction_id):
    user_id = session.get("user")
    if not user_id:
        return redirect(url_for("login"))

    response = (
        supabase
        .table("transactions")
        .select("id, description, type, amount, category")
        .eq("id", transaction_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    data = response.data
    if not data:
        return "Transaction not found or access denied", 404

    transaction = data[0]

    if request.method == "POST":
        supabase.table("transactions").update({
            "description": request.form["description"].strip(),
            "amount": float(request.form["amount"]),
            "type": request.form["type"],
            "category": request.form.get("category") or None
        }).eq("id", transaction_id).eq("user_id", user_id).execute()

        return redirect(url_for("home"))

    return render_template("edit_transaction.html", transaction=transaction)


@app.route('/add', methods=['GET', 'POST'])
def add_transaction():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    if request.method == 'POST':
        message = request.form.get('message', '')
        parsed = analyze_message(message) or []

        for t in parsed:
            if t.get('amount') and t.get('type') != 'unknown':
                supabase.table('transactions').insert({
                    "user_id": user_id,
                    "description": message,
                    "type": t['type'],
                    "amount": t['amount'],
                    "category": t.get('category')
                }).execute()

        return redirect(url_for('transactions'))

    return render_template('add.html')


@app.route('/transactions')
def transactions():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    res = supabase.table('transactions').select('*').eq('user_id', user_id).execute()
    return render_template('transactions.html', transactions=res.data or [])

@app.route("/add-manual", methods=["GET", "POST"])
def add_manual():
    user_id = session.get("user")
    if not user_id:
        return redirect(url_for("login"))

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        amount = request.form.get("amount", "").strip()
        txn_type = request.form.get("type")
        category = request.form.get("category") or None
        date = request.form.get("date")  # YYYY-MM-DD

        if not description or not amount or txn_type not in ("income", "expense"):
            return render_template(
                "add_manual.html",
                error="Please fill all required fields."
            )

        try:
            amount = float(amount)
        except ValueError:
            return render_template(
                "add_manual.html",
                error="Amount must be a number."
            )

        now = datetime.now(timezone.utc)

        insert_data = {
            "user_id": user_id,
            "description": description,
            "amount": amount,
            "type": txn_type,
            "category": category
        }

        if date:
            y, m, d = map(int, date.split("-"))
            insert_data["created_at"] = now.replace(
                year=y, month=m, day=d
            ).isoformat()
        else:
            insert_data["created_at"] = now.isoformat()

        supabase.table("transactions").insert(insert_data).execute()

        return redirect(url_for("transactions"))

    return render_template("add_manual.html")



@app.route('/predictions')
def predictions():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    res = supabase.table('transactions').select('*').eq('user_id', user_id).execute()
    df = pd.DataFrame(res.data or [])

    if not df.empty:
        df.rename(columns={'created_at': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.normalize()
        df.dropna(subset=['date', 'amount'], inplace=True)

    simple = predict_all_horizon(df)
    advanced = predict_all_horizons_multi(df)

    return render_template('predictions.html', simple=simple, advanced=advanced)


@app.route('/charts')
def charts():
    user_id = session.get('user')
    if not user_id:
        return redirect(url_for('login'))

    res = supabase.table('transactions').select(
        'created_at, type, amount, category'
    ).eq('user_id', user_id).execute()

    transactions = res.data or []
    df = pd.DataFrame(transactions)

    chart_data = {
        "expense_trend": {"labels": [], "values": []},
        "income_expense": {"labels": [], "income": [], "expense": []},
        "category": {"labels": [], "values": []},
        "balance": {"labels": [], "values": []},
    }

    if df.empty:
        return render_template('charts.html', chart_data=chart_data)

    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    df['month'] = df['created_at'].dt.to_period('M').astype(str)
    df['category'] = df['category'].fillna('Other')

    expense_trend = df[df['type'] == 'expense'].groupby('month', as_index=False)['amount'].sum()
    chart_data['expense_trend'] = {
        "labels": expense_trend['month'].tolist(),
        "values": expense_trend['amount'].tolist()
    }

    income_expense = (
        df.groupby(['month', 'type'], as_index=False)['amount']
        .sum()
        .pivot(index='month', columns='type', values='amount')
        .fillna(0)
        .reset_index()
    )

    chart_data['income_expense'] = {
        "labels": income_expense['month'].tolist(),
        "income": income_expense.get('income', pd.Series()).tolist(),
        "expense": income_expense.get('expense', pd.Series()).tolist()
    }

    category = df[df['type'] == 'expense'].groupby('category', as_index=False)['amount'].sum()
    chart_data['category'] = {
        "labels": category['category'].tolist(),
        "values": category['amount'].tolist()
    }

    df_sorted = df.sort_values('created_at')
    df_sorted['signed_amount'] = df_sorted['amount'].where(
        df_sorted['type'] == 'income', -df_sorted['amount']
    )
    df_sorted['balance'] = df_sorted['signed_amount'].cumsum()

    chart_data['balance'] = {
        "labels": df_sorted['created_at'].dt.strftime('%Y-%m-%d').tolist(),
        "values": df_sorted['balance'].tolist()
    }

    return render_template('charts.html', chart_data=chart_data)


if __name__ == '__main__':
    # if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    #     webbrowser.open_new("http://127.0.0.1:5000/")
    app.run(debug=True)
