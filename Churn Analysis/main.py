# ==========================================
# 1. IMPORTS
# ==========================================
import os
import random
import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display

# Machine Learning: Random Forest for high performance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reference Date
ANALYSIS_DATE = datetime.date(2025, 5, 26)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def setup_plotting_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

def process_arabic_text(text):
    if not text: return ""
    reshaped_text = arabic_reshaper.reshape(str(text))
    bidi_text = get_display(reshaped_text)
    return bidi_text

# ==========================================
# 3. CORE FUNCTIONS
# ==========================================

def generate_synthetic_data(n_customers: int = 10000) -> pd.DataFrame:
    from faker import Faker
    
    # Fixing the Random Seed for Reproducibility
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)
    
    logger.info(f"Generating data with stronger business rules for {n_customers} customers...")
    
    fake = Faker('ar_SA')
    CITIES = ["الرياض", "جدة", "مكة", "الدمام", "الخبر", "المدينة المنورة", "تبوك", "حائل", "بريدة", "عسير", "جيزان"]
    data = []

    for i in range(n_customers):
        registration_date = fake.date_between(start_date='-5y', end_date='-1y')
        
        is_high_risk = np.random.rand() < 0.3
        
        if not is_high_risk:
            last_purchase_date = fake.date_between(start_date='-6m', end_date='today')
            # Active customer: medium to high number of orders
            total_orders = np.random.randint(10, 50) 
        else:
            last_purchase_date = fake.date_between(start_date='-3y', end_date='-6m')
            # Churned customer: low number of orders
            total_orders = np.random.randint(1, 20) 
            
        if last_purchase_date < registration_date:
            last_purchase_date = registration_date + datetime.timedelta(days=np.random.randint(1, 30))

        total_spent = np.random.randint(100, 20000)
        if np.random.rand() < 0.05: total_spent = np.nan

        data.append({
            'customer_id': 1000 + i,
            'full_name': fake.name(),
            'city': random.choice(CITIES),
            'registration_date': registration_date,
            'last_purchase_date': last_purchase_date,
            'total_spent': total_spent,
            'total_orders': total_orders,
        })

    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data and feature engineering."""
    logger.info("Starting data preprocessing...")
    
    median_spent = df['total_spent'].median()
    df['total_spent'] = df['total_spent'].fillna(median_spent)
    
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    
    # Feature Engineering
    df['customer_seniority_days'] = (pd.to_datetime(ANALYSIS_DATE) - df['registration_date']).dt.days
    df['seniority_years'] = df['customer_seniority_days'] / 365.25
    df['days_since_last_purchase'] = (pd.to_datetime(ANALYSIS_DATE) - df['last_purchase_date']).dt.days
    df['avg_order_value'] = df['total_spent'] / df['total_orders'].replace(0, 1)
    
    # Target
    df['is_churned'] = (df['days_since_last_purchase'] > 180).astype(int)
    
    return df

def perform_eda(df: pd.DataFrame):
    logger.info("Starting EDA visualization...")
    
    # Arabic labels setup
    txt_active = process_arabic_text("نشط")
    txt_churned = process_arabic_text("متوقف")
    txt_status = process_arabic_text("حالة العميل")
    
    max_years = int(df['seniority_years'].max())
    year_ticks = range(max_years + 1)
    
    # -------------------------------------------------------
    # A. Churn Rate
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_churned', data=df, palette="viridis")
    plt.title(process_arabic_text('توزيع العملاء (نشط vs متوقف)'))
    plt.xlabel(txt_status)
    plt.ylabel(process_arabic_text('عدد العملاء'))
    plt.xticks([0, 1], [txt_active, txt_churned])
    plt.show()

    # -------------------------------------------------------
    # B. Value vs Frequency 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.boxplot(x='is_churned', y='total_orders', data=df, ax=ax1, palette="Set2")
    ax1.set_title(process_arabic_text('مقارنة إجمالي عدد الطلبات'))
    ax1.set_xlabel(txt_status)
    ax1.set_ylabel(process_arabic_text('إجمالي الطلبات'))
    ax1.set_xticklabels([txt_active, txt_churned])

    sns.boxplot(x='is_churned', y='avg_order_value', data=df, ax=ax2, palette="Set2")
    ax2.set_title(process_arabic_text('مقارنة متوسط قيمة الطلب'))
    ax2.set_xlabel(txt_status)
    ax2.set_ylabel(process_arabic_text('متوسط القيمة (ريال)'))
    ax2.set_xticklabels([txt_active, txt_churned])

    plt.suptitle(process_arabic_text('القيمة مقابل التكرار'))
    plt.show()

    # -------------------------------------------------------
    # C. Seniority Distribution 
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x='seniority_years',
        hue='is_churned', 
        multiple="stack",
        palette="crest", 
        edgecolor=".3", 
        linewidth=.5,
        bins=10
    )
    
    plt.title(process_arabic_text('توزيع أقدمية العملاء بالسنوات'))
    plt.xlabel(process_arabic_text('سنوات الخدمة (منذ التسجيل)'))
    plt.ylabel(process_arabic_text('عدد العملاء الحقيقي'))
    
    plt.xticks(year_ticks) 
    
    plt.legend(title=txt_status, labels=[txt_churned, txt_active])
    plt.show()

    # -------------------------------------------------------
    # D. Geographic Hotspots
    top_cities = df['city'].value_counts().head(10).index.tolist()
    df_cities = df[df['city'].isin(top_cities)]
    city_churn = df_cities.groupby('city')['is_churned'].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(12, 7))
    reshaped_labels = [process_arabic_text(city) for city in city_churn.index]
    sns.barplot(x=reshaped_labels, y=city_churn.values, palette="vlag")
    plt.title(process_arabic_text('نسبة التوقف في أهم 10 مدن'))
    plt.ylabel(process_arabic_text('نسبة التوقف (%)'))
    plt.xlabel(process_arabic_text('المدينة'))
    plt.show()

    # -------------------------------------------------------
    # E. Matrix (Spend vs Seniority) 
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(data=df, x='seniority_years', y='total_spent',
                         hue='is_churned', alpha=0.6, palette={0: 'blue', 1: 'red'})
    
    plt.xticks(year_ticks) 
    
    plt.axhline(df['total_spent'].median(), color='grey', linestyle='--')
    plt.axvline(df['seniority_years'].median(), color='grey', linestyle='--')
    
    plt.title(process_arabic_text('مصفوفة الإنفاق مقابل الأقدمية'))
    plt.xlabel(process_arabic_text('الأقدمية (سنوات)'))
    plt.ylabel(process_arabic_text('إجمالي الإنفاق (ريال)'))
    
    lgd = ax.get_legend()
    if lgd:
        lgd.set_title(txt_status)
        if len(lgd.texts) >= 2:
            lgd.texts[0].set_text(f"{txt_active} (0)")
            lgd.texts[1].set_text(f"{txt_churned} (1)")
    
    plt.show()

def train_churn_model(df: pd.DataFrame):
    logger.info("Adjusting Model for Class Imbalance...")

    features = ['customer_seniority_days', 'total_spent', 'total_orders', 'avg_order_value']
    target = 'is_churned'
    
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   class_weight='balanced') 
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logger.info(f"Model Training Completed.")
    print(f"\n{'-'*40}")
    print(f"Model Accuracy (Random Forest): {acc * 100:.2f}% (Check Recall!)")
    print(f"{'-'*40}")
    print("Classification Report (Adjusted for Imbalance):\n")
    print(classification_report(y_test, y_pred))

# ==========================================
# 4. MAIN 
# ==========================================
if __name__ == "__main__":
    setup_plotting_style()
    
    raw_df = generate_synthetic_data(n_customers=10000)
    clean_df = preprocess_data(raw_df)
    
    perform_eda(clean_df)
    train_churn_model(clean_df)