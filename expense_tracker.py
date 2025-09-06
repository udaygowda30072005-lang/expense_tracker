import os
import re
import json
import math
import pickle
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# Streamlit UI
import streamlit as st
from streamlit_option_menu import option_menu

# NLP & ML
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dates
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

# Security
import hashlib
import hmac

# Optional forecasting libs
try:
    from prophet import Prophet  # fbprophet renamed to prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

APP_TITLE = "Expense Tracker"
DB_FILE = "expenses.db"
MODEL_FILE = "category_model.pkl"
EXPORT_DIR = "exports"

DEFAULT_AI_KEYWORDS = {
    "food": ["restaurant", "cafe", "grocer", "food", "lunch", "dinner", "breakfast", "coffee", "swiggy", "zomato"],
    "transport": ["uber", "ola", "taxi", "bus", "train", "metro", "fuel", "diesel", "petrol", "gas", "parking", "transport", "bike", "car"],
    "entertainment": ["movie", "netflix", "concert", "game", "hobby", "entertainment", "spotify", "prime"],
    "shopping": ["amazon", "flipkart", "myntra", "target", "clothes", "shoes", "shopping", "electronics"],
    "utilities": ["electricity", "water", "internet", "broadband", "phone", "mobile", "recharge", "utility", "bill"],
    "health": ["doctor", "pharmacy", "chemist", "gym", "health", "medical", "medicine", "hospital"],
    "travel": ["hotel", "flight", "vacation", "travel", "airbnb", "train", "bus"],
    "education": ["course", "tuition", "udemy", "coursera", "book", "exam", "fee"],
    "rent": ["rent", "landlord", "apartment", "room"],
    "other": ["other", "misc"]
}

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£"
}

# ------------------------------
# Authentication Functions
# ------------------------------

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def init_auth_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, make_hashes(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username = ?", 
        (username,)
    )
    data = cur.fetchone()
    conn.close()
    
    if data and check_hashes(password, data[2]):
        return True
    return False

# ------------------------------
# Persistence Layer (SQLite)
# ------------------------------

def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db(user_id=None):
    conn = get_conn()
    cur = conn.cursor()
    
    # Add user_id column to expenses table if it doesn't exist
    try:
        cur.execute("ALTER TABLE expenses ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Add user_id column to budgets table if it doesn't exist
    try:
        cur.execute("ALTER TABLE budgets ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            user_id INTEGER
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT NOT NULL, -- 'YYYY-MM'
            category TEXT NOT NULL, -- 'ALL' for total budget
            amount REAL NOT NULL,
            user_id INTEGER
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_ts ON expenses(ts);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_cat ON expenses(category);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_expenses_user ON expenses(user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_budgets_user ON budgets(user_id);")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_budgets_month_cat_user ON budgets(month, category, user_id);")
    conn.commit()
    conn.close()

# ------------------------------
# Data Access Helpers (with user_id)
# ------------------------------

def insert_expense(ts: str, amount: float, category: str, description: str, user_id: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO expenses (ts, amount, category, description, user_id) VALUES (?, ?, ?, ?, ?)",
        (ts, amount, category, description, user_id)
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def delete_expense(expense_id: int, user_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM expenses WHERE id=? AND user_id=?", (expense_id, user_id))
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok

def fetch_expenses(user_id: int, category: Optional[str] = None, start: Optional[str] = None) -> pd.DataFrame:
    conn = get_conn()
    query = "SELECT id, ts, amount, category, description FROM expenses WHERE user_id=?"
    params: List = [user_id]
    
    if category and category.lower() != "all":
        query += " AND LOWER(category) = LOWER(?)"
        params.append(category)
    if start:
        query += " AND ts >= ?"
        params.append(start)
        
    query += " ORDER BY ts DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'])
    return df

def list_categories(user_id: int) -> List[str]:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT category FROM expenses WHERE user_id=? ORDER BY category", 
        conn, 
        params=(user_id,)
    )
    conn.close()
    cats = sorted(df['category'].tolist()) if not df.empty else []
    return cats

def upsert_budget(month: str, category: str, amount: float, user_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO budgets (month, category, amount, user_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(month, category, user_id) DO UPDATE SET amount=excluded.amount
        """,
        (month, category, amount, user_id)
    )
    conn.commit()
    conn.close()

def get_budgets(month: str, user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, month, category, amount FROM budgets WHERE month=? AND user_id=?", 
        conn, 
        params=(month, user_id)
    )
    conn.close()
    return df

def delete_budget(budget_id: int, user_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM budgets WHERE id=? AND user_id=?", (budget_id, user_id))
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok

# ------------------------------
# AI/NLP Helpers (unchanged)
# ------------------------------

def ai_keyword_category(description: str) -> Optional[str]:
    """Keyword-based categorization with fuzzy-ish matching."""
    if not description:
        return None
    text = description.lower()
    for cat, kws in DEFAULT_AI_KEYWORDS.items():
        for kw in kws:
            if kw in text:
                return cat.capitalize()
    return None

class CategoryModel:
    def __init__(self, model_path: str = MODEL_FILE):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self._load()
    
    def _load(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                try:
                    st.sidebar.success("ML model loaded successfully!")
                except Exception:
                    pass
            except Exception as e:
                try:
                    st.sidebar.warning(f"Failed to load model: {e}")
                except Exception:
                    pass
                self.pipeline = None
        else:
            try:
                st.sidebar.info("No trained model found. Add expenses and train the model.")
            except Exception:
                pass
            self.pipeline = None
                
    def train_if_possible(self, df: pd.DataFrame) -> Optional[str]:
        """Train when there are at least 40 labeled samples across 3+ classes."""
        msg = None
        if df.empty:
            return msg
            
        labeled = df.dropna(subset=['category'])
        if labeled.shape[0] < 40 or labeled['category'].nunique() < 3:
            return msg
            
        X = labeled['description'].fillna("")
        y = labeled['category']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        pipe = Pipeline(
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
            ('clf', LogisticRegression(max_iter=1000))
        )
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        # Save model
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(pipe, f)
            self.pipeline = pipe
            
            # Basic report
            report = classification_report(y_test, y_pred, output_dict=False)
            msg = "Category model trained. Classes: %d\n" % labeled['category'].nunique()
            msg += report
        except Exception as e:
            msg = f"Error saving model: {e}"
            
        return msg
        
    def predict(self, description: str) -> Optional[str]:
        if self.pipeline is None or not description:
            return None
        try:
            return str(self.pipeline.predict([description])[0])
        except Exception:
            return None

def parse_natural_expense(text: str, default_tz_now: Optional[dt.datetime] = None) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse amount, description (original text), and timestamp from natural text.
    Returns: (amount, description, ts_string)
    """
    if not default_tz_now:
        default_tz_now = dt.datetime.now()
    
    # Amount patterns: ₹120, Rs 120, 120.50, $20
    amt_match = re.search(r"(?:rs\.?|inr|₹|\$|eur|gbp)?\s*(\d+(?:[\.,]\d{1,2})?)", text, re.IGNORECASE)
    amount = None
    if amt_match:
        raw = amt_match.group(1).replace(',', '')
        try:
            amount = float(raw)
        except Exception:
            amount = None
            
    # Date detection with common phrases
    lower = text.lower()
    ts = default_tz_now
    
    if "yesterday" in lower:
        ts = ts - dt.timedelta(days=1)
    elif "today" in lower:
        pass
    elif "tomorrow" in lower:
        ts = ts + dt.timedelta(days=1)  # unlikely for expenses but supported
    else:
        # Try fuzzy parse for things like 'on 12 Aug', '15/08', 'last friday'
        try:
            parsed = date_parser.parse(text, fuzzy=True, dayfirst=True, default=default_tz_now)
            if parsed:
                ts = parsed
        except Exception:
            pass
            
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else None
    return amount, text.strip(), ts_str

def ai_categorize(description: str, model: CategoryModel) -> str:
    # 1) keyword
    cat = ai_keyword_category(description) if description else None
    
    # 2) ML model (only if pipeline exists)
    if not cat and getattr(model, 'pipeline', None) is not None:
        pred = model.predict(description or "")
        if pred:
            cat = pred
            
    # 3) sentiment fallback
    if not cat and description:
        try:
            sentiment = TextBlob(description).sentiment.polarity
            if sentiment > 0.2:
                cat = "Entertainment"
            elif sentiment < -0.2:
                cat = "Health"
        except Exception:
            pass
            
    # 4) default
    return (cat or "Other").capitalize()

# ------------------------------
# Analytics & Forecasting (with user_id)
# ------------------------------

def summary_stats(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            'total': 0.0,
            'count': 0,
            'average': 0.0,
            'by_category': {}
        }
        
    total = float(df['amount'].sum())
    count = int(df.shape[0])
    average = float(total / count) if count else 0.0
    by_category = df.groupby('category')['amount'].sum().sort_values(ascending=False).to_dict()
    
    return {
        'total': total,
        'count': count,
        'average': average,
        'by_category': by_category
    }

def monthly_totals(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return pd.DataFrame(columns=['month', 'amount'])
        
    s = df_all.copy()
    s['month'] = s['ts'].dt.to_period('M').astype(str)
    out = s.groupby('month', as_index=False)['amount'].sum().sort_values('month')
    return out

def forecast_spend(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Return DataFrame with columns [month, amount, type] and trend string."""
    hist = monthly_totals(df_all)
    if hist.shape[0] < 3:
        return pd.DataFrame(), "Need at least 3 months of data for forecast."

    # Calculate linear trend for messaging
    X = np.arange(hist.shape[0]).reshape(-1, 1)
    y = hist['amount'].values
    lin = LinearRegression().fit(X, y)
    trend = "increasing" if lin.coef_[0] > 0 else "decreasing"
    
    # Prefer Prophet, then SARIMAX, else linear extrapolation
    try:
        if _HAS_PROPHET:
            dfp = hist.rename(columns={'month': 'ds', 'amount': 'y'}).copy()
            # Convert ds to end of month dates
            dfp['ds'] = pd.to_datetime(dfp['ds']) + pd.offsets.MonthEnd(0)
            m = Prophet(seasonality_mode='additive', yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=3, freq='M')
            fcst = m.predict(future).tail(3)
            out_fc = pd.DataFrame({
                'month': fcst['ds'].dt.to_period('M').ast(str),
                'amount': fcst['yhat'],
                'type': 'forecast'
            })
        elif _HAS_STATSMODELS and hist.shape[0] >= 6:
            # Build a properly dated index
            month_index = pd.period_range(
                start=pd.Period(hist['month'].iloc[0], freq='M').to_timestamp(),
                periods=hist.shape[0],
                freq='M'
            )
            ts = pd.Series(hist['amount'].values, index=month_index)
            model = sm.tsa.statespace.SARIMAX(
                ts, order=(1,1,1), seasonal_order=(0,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False
            )
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=3)
            pred_mean = pred.predicted_mean
            out_fc = pd.DataFrame({
                'month': pred_mean.index.to_period('M').astype(str),
                'amount': pred_mean.values,
                'type': 'forecast'
            })
        else:
            future_idx = np.arange(hist.shape[0], hist.shape[0] + 3).reshape(-1, 1)
            preds = lin.predict(future_idx)
            last = pd.Period(hist['month'].iloc[-1]).to_timestamp() + pd.offsets.MonthBegin(1)
            months = [(last + pd.offsets.MonthBegin(i)).to_period('M').strftime('%Y-%m') for i in range(0, 3)]
            out_fc = pd.DataFrame({'month': months, 'amount': preds, 'type': 'forecast'})
    except Exception:
        # Fallback to linear
        future_idx = np.arange(hist.shape[0], hist.shape[0] + 3).reshape(-1, 1)
        preds = lin.predict(future_idx)
        last = pd.Period(hist['month'].iloc[-1]).to_timestamp() + pd.offsets.MonthBegin(1)
        months = [(last + pd.offsets.MonthBegin(i)).to_period('M').strftime('%Y-%m') for i in range(0, 3)]
        out_fc = pd.DataFrame({'month': months, 'amount': preds, 'type': 'forecast'})
        
    hist_df = hist.copy()
    hist_df['type'] = 'historical'
    return pd.concat([hist_df, out_fc], ignore_index=True), trend

def spending_insights(df_all: pd.DataFrame) -> Dict:
    if df_all.empty:
        return {}
        
    weekday_totals = df_all.groupby(df_all['ts'].dt.day_name())['amount'].sum()
    highest_day = weekday_totals.idxmax() if not weekday_totals.empty else None
    
    cat_totals = df_all.groupby('category')['amount'].sum()
    largest_cat = cat_totals.idxmax() if not cat_totals.empty else None
    
    # Recent vs older window
    recent = df_all.sort_values('ts', ascending=False).head(5)
    older = df_all.sort_values('ts', ascending=False).iloc[5:10]
    spending_increased = False
    if not recent.empty and not older.empty:
        if recent['amount'].sum() > older['amount'].sum() * 1.2:
            spending_increased = True
            
    # Large expenses via z-score
    amounts = df_all['amount']
    mean, std = amounts.mean(), amounts.std(ddof=0)
    z_large = df_all[amounts > (mean + 2 * (std if std > 0 else 1e-9))]
    
    return {
        'total_spent': float(amounts.sum()),
        'expense_count': int(df_all.shape[0]),
        'highest_day': highest_day or '—',
        'highest_day_amount': float(weekday_totals.max() if not weekday_totals.empty else 0.0),
        'largest_category': largest_cat or '—',
        'largest_category_amount': float(cat_totals.max() if not cat_totals.empty else 0.0),
        'spending_increased': spending_increased,
        'large_expenses_count': int(z_large.shape[0])
    }

# ------------------------------
# Budget utilities (with user_id)
# ------------------------------

def current_month_key(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now()
    return now.strftime('%Y-%m')

def compute_budget_status(df_month: pd.DataFrame, budgets_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Return (per-category status DF, totals dict)."""
    # Spent this month
    spent_cat = df_month.groupby('category')['amount'].sum() if not df_month.empty else pd.Series(dtype=float)
    spent_total = float(spent_cat.sum()) if not spent_cat.empty else 0.0
    
    # Budgets
    if budgets_df.empty:
        return pd.DataFrame(), {'spent_total': spent_total, 'budget_total': 0.0, 'remaining_total': -spent_total}
    
    # Per-category rows
    rows = []
    budget_total = 0.0
    
    for _, r in budgets_df.iterrows():
        cat = r['category']
        amt = float(r['amount'])
        
        if cat == 'ALL':
            budget_total += amt
            continue
            
        spent = float(spent_cat.get(cat, 0.0))
        remaining = amt - spent
        used_pct = (spent / amt * 100) if amt > 0 else 0.0
        rows.append({'category': cat, 'budget': amt, 'spent': spent, 'remaining': remaining, 'used_pct': used_pct})
    
    # Total budget row from 'ALL' or sum of category budgets
    if 'ALL' in budgets_df['category'].values:
        budget_total = float(budgets_df.loc[budgets_df['category'] == 'ALL', 'amount'].iloc[0])
    else:
        budget_total = sum([row['budget'] for row in rows])
        
    remaining_total = budget_total - spent_total
    
    status_df = pd.DataFrame(rows).sort_values('used_pct', ascending=False)
    return status_df, {'spent_total': spent_total, 'budget_total': budget_total, 'remaining_total': remaining_total}

# ------------------------------
# Export helpers (with user_id)
# ------------------------------

def safe_filename(base: str, ext: str) -> str:
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    base = re.sub(r"[^A-Za-z0-9_-]", "", base)[:40]
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR, exist_ok=True)
    return os.path.join(EXPORT_DIR, f"{base}{ts}.{ext}")

def export_csv(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    fn = safe_filename("expenses_", "csv")
    df.to_csv(fn, index=False)
    return fn

def export_excel(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    fn = safe_filename("expenses_", "xlsx")
    with pd.ExcelWriter(fn) as writer:
        df.to_excel(writer, index=False, sheet_name='Expenses')
    return fn

# ------------------------------
# Authentication UI
# ------------------------------

def login_page():
    st.title("Expense Tracker Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    # Get user ID
                    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
                    cur = conn.cursor()
                    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
                    user_id = cur.fetchone()[0]
                    conn.close()
                    
                    st.session_state.user_id = user_id
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif add_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

# ------------------------------
# Main App with Authentication
# ------------------------------

def main():
    st.set_page_config(
        page_title="Expense Tracker", 
        page_icon="💰", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Initialize databases
    init_auth_db()
    init_db()
    
    # Check if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        
    if not st.session_state.logged_in:
        login_page()
        return
        
    # User is logged in
    user_id = st.session_state.user_id
    username = st.session_state.username
    
    # Session state for currency and model
    if 'currency' not in st.session_state:
        st.session_state.currency = 'INR'
    if 'model' not in st.session_state:
        st.session_state.model = CategoryModel()
        
    currency = st.session_state.currency
    symbol = CURRENCY_SYMBOLS.get(currency, '₹')
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()
    
    st.sidebar.write(f"Logged in as: **{username}**")
    
    st.title(f"{username}'s Expense Tracker")
    st.caption("SQLite persistence • Budget alerts • ML categorization • Forecasting with Prophet/ARIMA fallback")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        currency = st.selectbox(
            "Currency", 
            list(CURRENCY_SYMBOLS.keys()), 
            index=list(CURRENCY_SYMBOLS.keys()).index(st.session_state.currency)
        )
        st.session_state.currency = currency
        symbol = CURRENCY_SYMBOLS[currency]
        
        menu = option_menu(
            "Go to", 
            ["Dashboard", "Add Expense", "View Expenses", "Analytics", "Budgets", "Export"], 
            icons=['house', 'plus-circle', 'list', 'graph-up', 'wallet', 'download'],
            menu_icon="cast",
            default_index=0
        )
        
        st.markdown("---")
        
        with st.expander("🔧 Train Category Model (optional)"):
            df_all_for_train = fetch_expenses(user_id)
            if st.button("Train/Refresh model"):
                msg = st.session_state.model.train_if_possible(df_all_for_train)
                if msg:
                    st.success("Model trained.")
                    st.text(msg)
                else:
                    st.info("Not enough labeled data to train. Add more expenses across categories.")
    
    # Helper to get filtered df
    def get_df(time_filter: str = "All", category_filter: str = "All") -> pd.DataFrame:
        start = None
        now = dt.datetime.now()
        if time_filter == 'week':
            start = (now - dt.timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
        elif time_filter == 'month':
            start = (now - dt.timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        elif time_filter == 'year':
            start = (now - dt.timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
        return fetch_expenses(user_id, category_filter, start)
    
    # ---------------- Dashboard ----------------
    if menu == "Dashboard":
        col1, col2, col3 = st.columns(3)
        df_all = fetch_expenses(user_id)
        summ_all = summary_stats(df_all)
        summ_month = summary_stats(get_df('month'))
        
        with col1:
            st.subheader("Total Spending")
            st.metric("All Time", f"{symbol}{summ_all['total']:.2f}")
        with col2:
            st.subheader("Monthly Spending")
            st.metric("Last 30 Days", f"{symbol}{summ_month['total']:.2f}")
        with col3:
            st.subheader("Expense Count")
            st.metric("Total Expenses", summ_all['count'])
            
        st.subheader("Recent Expenses")
        if not df_all.empty:
            st.dataframe(df_all[['ts', 'amount', 'category', 'description']].head(10))
        else:
            st.info("No expenses recorded yet.")
            
        # Budgets snapshot
        st.subheader("Budgets — This Month")
        mkey = current_month_key()
        budgets_df = get_budgets(mkey, user_id)
        month_df = get_df('month')
        status_df, totals = compute_budget_status(month_df, budgets_df)
        
        if not budgets_df.empty:
            colA, colB, colC = st.columns(3)
            colA.metric("Budget (Total)", f"{symbol}{totals['budget_total']:.2f}")
            colB.metric("Spent (Total)", f"{symbol}{totals['spent_total']:.2f}")
            remaining_str = f"{symbol}{totals['remaining_total']:.2f}"
            colC.metric("Remaining (Total)", remaining_str, delta=None)
            
            if not status_df.empty:
                bar = px.bar(
                    status_df, x='category', y='used_pct', 
                    title='Budget usage by category (%)',
                    labels={'used_pct': 'Used %', 'category': 'Category'}
                )
                st.plotly_chart(bar, use_container_width=True)
                
            if totals['remaining_total'] < 0:
                st.error("⚠ You have exceeded your total monthly budget!")
            elif totals['spent_total'] > 0 and totals['budget_total'] > 0 and totals['spent_total']/max(1, totals['budget_total']) > 0.9:
                st.warning("⚠ You have reached 90% of your monthly budget.")
        else:
            st.info("No budgets set for this month. Go to the Budgets tab to set them.")
            
        # Category pie
        st.subheader("Spending by Category (All Time)")
        if summ_all['by_category']:
            fig = px.pie(
                values=list(summ_all['by_category'].values()), 
                names=list(summ_all['by_category'].keys()), 
                title="Expense Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available.")
            
    # ---------------- Add Expense ----------------
    elif menu == "Add Expense":
        st.header("Add New Expense")
        tab1, tab2 = st.tabs(["Standard Entry", "Natural Language"])
        
        with tab1:
            with st.form("add_expense_form"):
                col1, col2 = st.columns(2)
                with col1:
                    amount = st.number_input(f"Amount ({symbol})", min_value=0.01, step=0.01, format="%.2f")
                    categories = ["Auto-Detect"] + (list_categories(user_id) or ["Food", "Transport", "Shopping", "Other"])
                    category = st.selectbox("Category", categories, index=0)
                with col2:
                    description = st.text_input("Description")
                    date_val = st.date_input("Date", dt.date.today())
                submitted = st.form_submit_button("Add Expense")
                
                if submitted:
                    # AI categorize if needed
                    chosen_cat = "" if category == "Auto-Detect" else category
                    if not chosen_cat.strip():
                        chosen_cat = ai_categorize(description, st.session_state.model)
                        
                    ts_str = dt.datetime.combine(date_val, dt.datetime.now().time()).strftime("%Y-%m-%d %H:%M:%S")
                    new_id = insert_expense(ts_str, float(amount), chosen_cat, description.strip(), user_id)
                    st.success(f"Expense added successfully! (ID: {new_id})")
                    
        with tab2:
            st.write("Describe your expense in natural language:")
            st.info("Examples: 'Spent ₹250 on groceries yesterday', 'Paid 1200 internet bill on 15 Aug'")
            natural_input = st.text_input("Expense description")
            
            if st.button("Add Expense from Text"):
                if natural_input.strip():
                    amt, desc, ts = parse_natural_expense(natural_input)
                    if not amt:
                        st.error("Could not detect amount. Please include a number like 250 or ₹250.")
                    else:
                        cat = ai_categorize(desc, st.session_state.model)
                        new_id = insert_expense(ts, amt, cat, desc, user_id)
                        st.success(f"Expense added successfully! (ID: {new_id})")
                else:
                    st.error("Please enter a description")
                    
    # ---------------- View Expenses ----------------
    elif menu == "View Expenses":
        st.header("View Expenses")
        col1, col2 = st.columns(2)
        with col1:
            categories = ["All"] + (list_categories(user_id) or [])
            category_filter = st.selectbox("Filter by Category", categories)
        with col2:
            time_options = ["All", "week", "month", "year"]
            time_filter = st.selectbox("Filter by Time", time_options)
            
        df = get_df(time_filter, category_filter)
        if not df.empty:
            st.dataframe(df)
            
            st.subheader("Delete Expense")
            exp_id = st.number_input("Enter Expense ID to delete", min_value=1, step=1)
            if st.button("Delete Expense"):
                ok = delete_expense(int(exp_id), user_id)
                if ok:
                    st.success(f"Expense ID {int(exp_id)} deleted successfully!")
                    st.rerun()
                else:
                    st.error("Expense not found.")
        else:
            st.info("No expenses found with the selected filters.")
            
    # ---------------- Analytics ----------------
    elif menu == "Analytics":
        st.header("Spending Analytics")
        tab1, tab2, tab3 = st.tabs(["Summary", "Forecast", "Insights"])
        
        with tab1:
            st.subheader("Expense Summary")
            time_options = ["All", "week", "month", "year"]
            period = st.selectbox("Summary Period", time_options)
            df = get_df(period)
            summ = summary_stats(df)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", f"{symbol}{summ['total']:.2f}")
            col2.metric("Number of Expenses", summ['count'])
            col3.metric("Average Expense", f"{symbol}{summ['average']:.2f}")
            
            st.subheader("Spending by Category")
            if summ['by_category']:
                fig = px.bar(
                    x=list(summ['by_category'].keys()), 
                    y=list(summ['by_category'].values()), 
                    title="Expenses by Category",
                    labels={'x': 'Category', 'y': f'Amount ({symbol})'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No expenses to summarize.")
                
        with tab2:
            st.subheader("Spending Forecast (next 3 months)")
            df_all = fetch_expenses(user_id)
            fc_df, trend = forecast_spend(df_all)
            
            if not fc_df.empty:
                fig = go.Figure()
                hist = fc_df[fc_df['type'] == 'historical']
                fore = fc_df[fc_df['type'] == 'forecast']
                
                fig.add_trace(go.Scatter(
                    x=hist['month'], y=hist['amount'],
                    mode='lines+markers', name='Historical'
                ))
                fig.add_trace(go.Scatter(
                    x=fore['month'], y=fore['amount'],
                    mode='lines+markers', name='Forecast',
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title="Monthly Spending Forecast",
                    xaxis_title="Month",
                    yaxis_title=f"Amount ({symbol})",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Your spending is generally {trend} over time.")
            else:
                st.info(trend)
                
        with tab3:
            st.subheader("Spending Insights")
            df_all = fetch_expenses(user_id)
            ins = spending_insights(df_all)
            
            if ins:
                col1, col2 = st.columns(2)
                col1.metric("Total Spent", f"{symbol}{ins['total_spent']:.2f}")
                col1.metric("Expense Count", ins['expense_count'])
                col1.metric("Highest Spending Day", ins['highest_day'])
                
                col2.metric("Largest Category", ins['largest_category'])
                avg = (ins['total_spent']/ins['expense_count']) if ins['expense_count'] else 0.0
                col2.metric("Average Expense", f"{symbol}{avg:.2f}")
                col2.metric("Large Expenses (z>2)", ins['large_expenses_count'])
                
                if ins['spending_increased']:
                    st.warning("⚠ Your spending has increased recently. Consider reviewing your expenses.")
                if ins['large_expenses_count'] > 0:
                    st.info("💡 You have unusually large expenses. Review these for potential savings.")
            else:
                st.info("No insights available. Add more expenses to generate insights.")
                
    # ---------------- Budgets ----------------
    elif menu == "Budgets":
        st.header("Monthly Budgets")
        mkey = current_month_key()
        st.write(f"Current month: *{mkey}*")
        budgets_df = get_budgets(mkey, user_id)
        
        # Add/Edit Budget Form
        with st.form("add_budget_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Category", ["ALL", "Food", "Transport", "Shopping", "Utilities", "Health", "Travel", "Education", "Rent", "Entertainment", "Other"])
            with col2:
                amount = st.number_input(f"Budget Amount ({symbol})", min_value=0.0, step=100.0)
            with col3:
                submit = st.form_submit_button("Save Budget")
                
            if submit:
                upsert_budget(mkey, category, float(amount), user_id)
                st.success("Budget saved.")
                st.rerun()
        
        # Display and Delete Existing Budgets
        st.subheader("Existing Budgets")
        if not budgets_df.empty:
            for _, budget in budgets_df.iterrows():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    st.write(f"**{budget['category']}**")
                with col2:
                    st.write(f"{symbol}{budget['amount']:.2f}")
                with col3:
                    if st.button("Edit", key=f"edit_{budget['id']}"):
                        st.session_state.editing_budget = budget['id']
                        st.session_state.edit_category = budget['category']
                        st.session_state.edit_amount = budget['amount']
                with col4:
                    if st.button("Delete", key=f"delete_{budget['id']}"):
                        if delete_budget(budget['id'], user_id):
                            st.success("Budget deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete budget.")
            
            # Edit Budget Form (appears when Edit button is clicked)
            if 'editing_budget' in st.session_state:
                st.subheader("Edit Budget")
                with st.form("edit_budget_form"):
                    edit_category = st.text_input("Category", value=st.session_state.edit_category, disabled=True)
                    edit_amount = st.number_input(f"Amount ({symbol})", value=float(st.session_state.edit_amount), min_value=0.0, step=100.0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Update Budget"):
                            upsert_budget(mkey, edit_category, edit_amount, user_id)
                            del st.session_state.editing_budget
                            st.success("Budget updated!")
                            st.rerun()
                    with col2:
                        if st.form_submit_button("Cancel"):
                            del st.session_state.editing_budget
                            st.rerun()
        else:
            st.info("No budgets set for this month.")
            
        # Show budget status
        df_month = fetch_expenses(user_id, start=dt.datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S'))
        status_df, totals = compute_budget_status(df_month, budgets_df)
        if not status_df.empty:
            st.subheader("Per-Category Status")
            st.dataframe(status_df)
            
    # ---------------- Export ----------------
    elif menu == "Export":
        st.header("Export Data")
        df_all = fetch_expenses(user_id)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export CSV"):
                fn = export_csv(df_all)
                if fn:
                    with open(fn, 'rb') as f:
                        st.download_button(
                            "Download CSV", f, 
                            file_name=os.path.basename(fn),
                            mime="text/csv"
                        )
                else:
                    st.error("Nothing to export.")
                    
        with col2:
            if st.button("Export Excel"):
                fn = export_excel(df_all)
                if fn:
                    with open(fn, 'rb') as f:
                        st.download_button(
                            "Download Excel", f, 
                            file_name=os.path.basename(fn),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("Nothing to export.")

if __name__ == "__main__":
    main()