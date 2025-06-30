"""
Advanced Feature Engineering Pipeline for CLV Prediction

This module creates sophisticated features from raw customer and transaction data
to improve CLV prediction model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')


class CLVFeatureEngineering:
    """Advanced feature engineering for Customer Lifetime Value prediction."""
    
    def __init__(self, prediction_horizon_days: int = 365):
        """
        Initialize feature engineering pipeline.
        
        Args:
            prediction_horizon_days: Number of days to predict CLV for
        """
        self.prediction_horizon_days = prediction_horizon_days
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features from date columns."""
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df[f'{date_col}_year'] = df[date_col].dt.year
        df[f'{date_col}_month'] = df[date_col].dt.month
        df[f'{date_col}_quarter'] = df[date_col].dt.quarter
        df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
        df[f'{date_col}_week'] = df[date_col].dt.isocalendar().week
        
        # Season features
        df[f'{date_col}_season'] = df[f'{date_col}_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Holiday indicators
        df[f'{date_col}_is_holiday_season'] = df[f'{date_col}_month'].isin([11, 12]).astype(int)
        df[f'{date_col}_is_weekend'] = df[f'{date_col}_dayofweek'].isin([5, 6]).astype(int)
        
        return df
    
    def create_rfm_features(self, transactions_df: pd.DataFrame, 
                           customer_df: pd.DataFrame, 
                           analysis_date: datetime = None) -> pd.DataFrame:
        """Create Recency, Frequency, Monetary (RFM) features."""
        
        if analysis_date is None:
            analysis_date = transactions_df['transaction_date'].max()
        
        # Aggregate transaction data by customer
        rfm_data = []
        
        for customer_id in customer_df['customer_id'].unique():
            customer_transactions = transactions_df[
                transactions_df['customer_id'] == customer_id
            ].copy()
            
            if len(customer_transactions) == 0:
                # Handle customers with no transactions
                rfm_data.append({
                    'customer_id': customer_id,
                    'recency_days': 9999,
                    'frequency': 0,
                    'monetary_total': 0,
                    'monetary_avg': 0
                })
                continue
            
            # Recency: days since last purchase
            last_purchase = customer_transactions['transaction_date'].max()
            recency = (analysis_date - last_purchase).days
            
            # Frequency: number of unique purchase dates
            frequency = customer_transactions['transaction_date'].nunique()
            
            # Monetary: total and average spending
            monetary_total = customer_transactions['final_price'].sum()
            monetary_avg = customer_transactions['final_price'].mean()
            
            rfm_data.append({
                'customer_id': customer_id,
                'recency_days': recency,
                'frequency': frequency,
                'monetary_total': monetary_total,
                'monetary_avg': monetary_avg
            })
        
        rfm_df = pd.DataFrame(rfm_data)
        
        # Create RFM scores (1-5 scale)
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency_days'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary_total'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        rfm_df['recency_score'] = pd.to_numeric(rfm_df['recency_score'], errors='coerce').fillna(1)
        rfm_df['frequency_score'] = pd.to_numeric(rfm_df['frequency_score'], errors='coerce').fillna(1)
        rfm_df['monetary_score'] = pd.to_numeric(rfm_df['monetary_score'], errors='coerce').fillna(1)
        
        # Combined RFM score
        rfm_df['rfm_score'] = (
            rfm_df['recency_score'] * 100 + 
            rfm_df['frequency_score'] * 10 + 
            rfm_df['monetary_score']
        )
        
        return rfm_df
    
    def create_behavioral_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced behavioral features from transaction history."""
        
        behavioral_features = []
        
        for customer_id in transactions_df['customer_id'].unique():
            customer_transactions = transactions_df[
                transactions_df['customer_id'] == customer_id
            ].copy()
            
            if len(customer_transactions) == 0:
                continue
            
            customer_transactions = customer_transactions.sort_values('transaction_date')
            
            # Time-based features
            first_purchase = customer_transactions['transaction_date'].min()
            last_purchase = customer_transactions['transaction_date'].max()
            customer_lifetime_days = (last_purchase - first_purchase).days + 1
            
            # Purchase patterns
            unique_purchase_dates = customer_transactions['transaction_date'].nunique()
            total_transactions = len(customer_transactions)
            
            # Calculate days between purchases
            purchase_dates = customer_transactions['transaction_date'].unique()
            if len(purchase_dates) > 1:
                purchase_dates = pd.to_datetime(purchase_dates)
                purchase_dates = np.sort(purchase_dates)
                days_between = np.diff(purchase_dates).astype('timedelta64[D]').astype(int)
                avg_days_between_purchases = np.mean(days_between)
                std_days_between_purchases = np.std(days_between)
            else:
                avg_days_between_purchases = customer_lifetime_days
                std_days_between_purchases = 0
            
            # Product category diversity
            unique_categories = customer_transactions['product_category'].nunique()
            total_categories = customer_transactions['product_category'].count()
            category_diversity = unique_categories / max(total_categories, 1)
            
            # Discount sensitivity
            discount_transactions = customer_transactions[customer_transactions['discount'] > 0]
            discount_sensitivity = len(discount_transactions) / max(len(customer_transactions), 1)
            avg_discount_taken = customer_transactions['discount'].mean()
            
            # Spending patterns
            total_spent = customer_transactions['final_price'].sum()
            avg_transaction_value = customer_transactions['final_price'].mean()
            std_transaction_value = customer_transactions['final_price'].std()
            max_transaction_value = customer_transactions['final_price'].max()
            min_transaction_value = customer_transactions['final_price'].min()
            
            # Coefficient of variation for spending
            cv_spending = std_transaction_value / max(avg_transaction_value, 0.01)
            
            # Trend analysis (if enough data)
            if len(customer_transactions) >= 3:
                # Simple trend: compare first half vs second half spending
                mid_point = len(customer_transactions) // 2
                first_half_avg = customer_transactions.iloc[:mid_point]['final_price'].mean()
                second_half_avg = customer_transactions.iloc[mid_point:]['final_price'].mean()
                spending_trend = (second_half_avg - first_half_avg) / max(first_half_avg, 0.01)
            else:
                spending_trend = 0
            
            # Seasonal preferences
            monthly_spending = customer_transactions.groupby(
                customer_transactions['transaction_date'].dt.month
            )['final_price'].sum()
            seasonal_concentration = monthly_spending.std() / max(monthly_spending.mean(), 0.01)
            
            behavioral_features.append({
                'customer_id': customer_id,
                'customer_lifetime_days': customer_lifetime_days,
                'unique_purchase_dates': unique_purchase_dates,
                'total_transactions': total_transactions,
                'avg_days_between_purchases': avg_days_between_purchases,
                'std_days_between_purchases': std_days_between_purchases,
                'category_diversity': category_diversity,
                'unique_categories': unique_categories,
                'discount_sensitivity': discount_sensitivity,
                'avg_discount_taken': avg_discount_taken,
                'total_spent': total_spent,
                'avg_transaction_value': avg_transaction_value,
                'std_transaction_value': std_transaction_value,
                'max_transaction_value': max_transaction_value,
                'min_transaction_value': min_transaction_value,
                'cv_spending': cv_spending,
                'spending_trend': spending_trend,
                'seasonal_concentration': seasonal_concentration
            })
        
        return pd.DataFrame(behavioral_features)
    
    def create_demographic_features(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Create and encode demographic features."""
        
        df = customers_df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 25, 35, 45, 55, 100], 
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        
        # Income groups
        df['income_group'] = pd.cut(
            df['income'],
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
        )
        
        # Customer tenure (days since registration)
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        current_date = datetime.now()
        df['tenure_days'] = (current_date - df['registration_date']).dt.days
        
        # Tenure groups
        df['tenure_group'] = pd.cut(
            df['tenure_days'],
            bins=[0, 90, 365, 730, float('inf')],
            labels=['New', 'Recent', 'Established', 'Veteran']
        )
        
        return df
    
    def create_campaign_features(self, campaigns_df: pd.DataFrame, 
                               customers_df: pd.DataFrame) -> pd.DataFrame:
        """Create marketing campaign response features."""
        
        campaign_features = []
        
        for customer_id in customers_df['customer_id'].unique():
            customer_campaigns = campaigns_df[
                campaigns_df['customer_id'] == customer_id
            ]
            
            if len(customer_campaigns) == 0:
                campaign_features.append({
                    'customer_id': customer_id,
                    'total_campaigns': 0,
                    'campaigns_responded': 0,
                    'campaign_response_rate': 0,
                    'email_campaigns': 0,
                    'sms_campaigns': 0,
                    'push_campaigns': 0,
                    'social_campaigns': 0
                })
                continue
            
            total_campaigns = len(customer_campaigns)
            campaigns_responded = customer_campaigns['responded'].sum()
            response_rate = campaigns_responded / max(total_campaigns, 1)
            
            # Channel breakdown
            channel_counts = customer_campaigns['channel'].value_counts()
            
            campaign_features.append({
                'customer_id': customer_id,
                'total_campaigns': total_campaigns,
                'campaigns_responded': campaigns_responded,
                'campaign_response_rate': response_rate,
                'email_campaigns': channel_counts.get('Email', 0),
                'sms_campaigns': channel_counts.get('SMS', 0),
                'push_campaigns': channel_counts.get('Push', 0),
                'social_campaigns': channel_counts.get('Social', 0)
            })
        
        return pd.DataFrame(campaign_features)
    
    def calculate_clv_target(self, transactions_df: pd.DataFrame, 
                           customers_df: pd.DataFrame,
                           prediction_start_date: datetime = None) -> pd.DataFrame:
        """Calculate CLV target variable for the prediction horizon."""
        
        if prediction_start_date is None:
            # Use 80% of the data timeline as prediction start
            min_date = transactions_df['transaction_date'].min()
            max_date = transactions_df['transaction_date'].max()
            timeline = (max_date - min_date).days
            prediction_start_date = min_date + timedelta(days=int(timeline * 0.8))
        
        prediction_end_date = prediction_start_date + timedelta(days=self.prediction_horizon_days)
        
        clv_targets = []
        
        for customer_id in customers_df['customer_id'].unique():
            # Get transactions in the prediction window
            future_transactions = transactions_df[
                (transactions_df['customer_id'] == customer_id) &
                (transactions_df['transaction_date'] >= prediction_start_date) &
                (transactions_df['transaction_date'] <= prediction_end_date)
            ]
            
            clv = future_transactions['final_price'].sum()
            
            clv_targets.append({
                'customer_id': customer_id,
                'clv_target': clv
            })
        
        return pd.DataFrame(clv_targets)
    
    def build_feature_matrix(self, customers_df: pd.DataFrame, 
                           transactions_df: pd.DataFrame,
                           campaigns_df: pd.DataFrame,
                           feature_selection_k: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build the complete feature matrix for CLV prediction."""
        
        print("Creating temporal features...")
        customers_with_temporal = self.create_temporal_features(
            customers_df, 'registration_date'
        )
        
        print("Creating RFM features...")
        rfm_features = self.create_rfm_features(transactions_df, customers_df)
        
        print("Creating behavioral features...")
        behavioral_features = self.create_behavioral_features(transactions_df)
        
        print("Creating demographic features...")
        demographic_features = self.create_demographic_features(customers_with_temporal)
        
        print("Creating campaign features...")
        campaign_features = self.create_campaign_features(campaigns_df, customers_df)
        
        print("Calculating CLV targets...")
        clv_targets = self.calculate_clv_target(transactions_df, customers_df)
        
        # Merge all features
        print("Merging features...")
        feature_matrix = demographic_features.merge(
            rfm_features, on='customer_id', how='left'
        ).merge(
            behavioral_features, on='customer_id', how='left'
        ).merge(
            campaign_features, on='customer_id', how='left'
        ).merge(
            clv_targets, on='customer_id', how='left'
        )
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(0)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        categorical_columns = [
            'gender', 'city_tier', 'marketing_channel', 'segment',
            'age_group', 'income_group', 'tenure_group', 'registration_date_season'
        ]
        
        for col in categorical_columns:
            if col in feature_matrix.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    feature_matrix[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        feature_matrix[col].astype(str)
                    )
                else:
                    feature_matrix[f'{col}_encoded'] = self.encoders[col].transform(
                        feature_matrix[col].astype(str)
                    )
        
        # Select features (exclude target and ID columns)
        feature_columns = [col for col in feature_matrix.columns 
                          if col not in ['customer_id', 'clv_target', 'registration_date'] 
                          and not col.endswith('_group') 
                          and col not in categorical_columns]
        
        X = feature_matrix[feature_columns]
        y = feature_matrix[['customer_id', 'clv_target']]
        
        # Feature selection
        if feature_selection_k and feature_selection_k < len(feature_columns):
            print(f"Selecting top {feature_selection_k} features...")
            selector = SelectKBest(score_func=f_regression, k=feature_selection_k)
            X_selected = selector.fit_transform(X, feature_matrix['clv_target'])
            
            # Get selected feature names
            selected_features = np.array(feature_columns)[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Selected features: {len(self.feature_names)}")
        
        return X, y


def main():
    """Test the feature engineering pipeline."""
    
    print("Loading data...")
    customers_df = pd.read_csv('data/raw/customers.csv')
    transactions_df = pd.read_csv('data/raw/transactions.csv')
    campaigns_df = pd.read_csv('data/raw/campaigns.csv')
    
    # Convert date columns
    customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    
    print("Building feature matrix...")
    feature_engineer = CLVFeatureEngineering(prediction_horizon_days=365)
    
    X, y = feature_engineer.build_feature_matrix(
        customers_df, transactions_df, campaigns_df
    )
    
    # Save processed features
    print("Saving processed features...")
    feature_data = X.copy()
    feature_data['customer_id'] = y['customer_id']
    feature_data['clv_target'] = y['clv_target']
    
    feature_data.to_csv('data/processed/feature_matrix.csv', index=False)
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        for feature in feature_engineer.feature_names:
            f.write(f"{feature}\n")
    
    print(f"\nFeature engineering complete!")
    print(f"Features saved: {X.shape[1]} features for {X.shape[0]} customers")
    print(f"Target statistics:")
    print(f"  Mean CLV: ${y['clv_target'].mean():.2f}")
    print(f"  Median CLV: ${y['clv_target'].median():.2f}")
    print(f"  Max CLV: ${y['clv_target'].max():.2f}")


if __name__ == "__main__":
    main()