"""
Synthetic E-commerce Dataset Generator for CLV Prediction

This module generates realistic e-commerce customer transaction data
for training and testing CLV prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, List
import json


class EcommerceDataGenerator:
    """Generate synthetic e-commerce customer transaction data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Product categories and their average prices
        self.product_categories = {
            'Electronics': {'avg_price': 299.99, 'std': 150.0},
            'Clothing': {'avg_price': 49.99, 'std': 25.0},
            'Home & Garden': {'avg_price': 79.99, 'std': 40.0},
            'Books': {'avg_price': 19.99, 'std': 10.0},
            'Sports': {'avg_price': 89.99, 'std': 45.0},
            'Beauty': {'avg_price': 34.99, 'std': 20.0},
            'Toys': {'avg_price': 24.99, 'std': 15.0},
            'Automotive': {'avg_price': 149.99, 'std': 75.0}
        }
        
        # Customer segments with different behaviors
        self.customer_segments = {
            'High Value': {
                'probability': 0.15,
                'purchase_frequency': 8.5,
                'avg_order_value': 180.0,
                'churn_rate': 0.05
            },
            'Medium Value': {
                'probability': 0.35,
                'purchase_frequency': 4.2,
                'avg_order_value': 85.0,
                'churn_rate': 0.15
            },
            'Low Value': {
                'probability': 0.35,
                'purchase_frequency': 2.1,
                'avg_order_value': 45.0,
                'churn_rate': 0.25
            },
            'New Customer': {
                'probability': 0.15,
                'purchase_frequency': 1.5,
                'avg_order_value': 65.0,
                'churn_rate': 0.40
            }
        }
    
    def generate_customers(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate customer demographic data."""
        
        customers = []
        
        for customer_id in range(1, n_customers + 1):
            # Assign customer segment
            segment = np.random.choice(
                list(self.customer_segments.keys()),
                p=[seg['probability'] for seg in self.customer_segments.values()]
            )
            
            # Generate demographics based on segment
            if segment == 'High Value':
                age = int(np.random.normal(45, 12))
                income = int(np.random.normal(85000, 25000))
            elif segment == 'Medium Value':
                age = int(np.random.normal(38, 15))
                income = int(np.random.normal(55000, 20000))
            elif segment == 'Low Value':
                age = int(np.random.normal(32, 18))
                income = int(np.random.normal(35000, 15000))
            else:  # New Customer
                age = int(np.random.normal(28, 12))
                income = int(np.random.normal(45000, 20000))
            
            # Ensure realistic bounds
            age = max(18, min(80, age))
            income = max(20000, min(200000, income))
            
            # Registration date (last 3 years)
            registration_date = datetime.now() - timedelta(
                days=np.random.randint(1, 1095)
            )
            
            customers.append({
                'customer_id': customer_id,
                'segment': segment,
                'age': age,
                'income': income,
                'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
                'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], p=[0.3, 0.45, 0.25]),
                'registration_date': registration_date,
                'marketing_channel': np.random.choice([
                    'Organic', 'Social Media', 'Email', 'Paid Search', 'Referral'
                ], p=[0.25, 0.20, 0.15, 0.25, 0.15])
            })
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df: pd.DataFrame, 
                            end_date: datetime = None) -> pd.DataFrame:
        """Generate transaction data for customers."""
        
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            segment = customer['segment']
            registration_date = customer['registration_date']
            
            # Customer behavior parameters
            segment_info = self.customer_segments[segment]
            annual_purchases = np.random.poisson(segment_info['purchase_frequency'])
            
            # Generate purchases over time
            current_date = registration_date
            
            # Simulate customer lifecycle
            is_churned = False
            churn_date = None
            
            # Determine if/when customer churns
            if np.random.random() < segment_info['churn_rate']:
                days_active = np.random.exponential(365)  # Average 1 year before churn
                churn_date = registration_date + timedelta(days=days_active)
                is_churned = True
            
            purchase_count = 0
            while current_date < end_date and purchase_count < annual_purchases * 2:
                # Stop if customer has churned
                if is_churned and current_date > churn_date:
                    break
                
                # Time between purchases (exponential distribution)
                days_between = max(1, int(np.random.exponential(365 / segment_info['purchase_frequency'])))
                current_date += timedelta(days=days_between)
                
                if current_date > end_date:
                    break
                
                # Generate order value
                base_value = segment_info['avg_order_value']
                order_value = max(10, np.random.lognormal(
                    np.log(base_value), 0.5
                ))
                
                # Seasonal effects
                month = current_date.month
                seasonal_multiplier = 1.0
                if month in [11, 12]:  # Holiday season
                    seasonal_multiplier = 1.3
                elif month in [6, 7, 8]:  # Summer
                    seasonal_multiplier = 1.1
                
                order_value *= seasonal_multiplier
                
                # Number of items in order
                n_items = max(1, int(np.random.poisson(order_value / 50)))
                
                # Generate items
                for item_num in range(n_items):
                    category = np.random.choice(list(self.product_categories.keys()))
                    cat_info = self.product_categories[category]
                    
                    item_price = max(5, np.random.normal(
                        cat_info['avg_price'], cat_info['std']
                    ))
                    
                    # Discount probability based on customer segment
                    discount = 0
                    if segment == 'High Value' and np.random.random() < 0.15:
                        discount = np.random.uniform(0.05, 0.15)
                    elif np.random.random() < 0.08:
                        discount = np.random.uniform(0.05, 0.25)
                    
                    final_price = item_price * (1 - discount)
                    
                    transactions.append({
                        'transaction_id': transaction_id,
                        'customer_id': customer_id,
                        'transaction_date': current_date,
                        'product_category': category,
                        'item_price': round(item_price, 2),
                        'discount': round(discount, 3),
                        'final_price': round(final_price, 2),
                        'quantity': 1  # Simplify to 1 item per row
                    })
                    
                    transaction_id += 1
                
                purchase_count += 1
        
        return pd.DataFrame(transactions)
    
    def generate_complete_dataset(self, n_customers: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete customer and transaction datasets."""
        
        print(f"Generating {n_customers} customers...")
        customers_df = self.generate_customers(n_customers)
        
        print("Generating transactions...")
        transactions_df = self.generate_transactions(customers_df)
        
        print(f"Generated {len(transactions_df)} transactions")
        
        return customers_df, transactions_df
    
    def add_marketing_campaigns(self, customers_df: pd.DataFrame, 
                              transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Add marketing campaign data."""
        
        campaigns = []
        
        # Define campaign types
        campaign_types = [
            {'name': 'Summer Sale', 'start': '2023-06-01', 'end': '2023-08-31', 'response_rate': 0.12},
            {'name': 'Black Friday', 'start': '2023-11-20', 'end': '2023-11-30', 'response_rate': 0.25},
            {'name': 'New Year', 'start': '2024-01-01', 'end': '2024-01-15', 'response_rate': 0.08},
            {'name': 'Spring Collection', 'start': '2024-03-01', 'end': '2024-05-31', 'response_rate': 0.10}
        ]
        
        for campaign in campaign_types:
            # Select customers for campaign (higher probability for high-value customers)
            campaign_customers = customers_df.sample(
                frac=0.6, 
                weights=customers_df['segment'].map({
                    'High Value': 0.8, 'Medium Value': 0.6, 'Low Value': 0.4, 'New Customer': 0.3
                })
            )
            
            for _, customer in campaign_customers.iterrows():
                responded = np.random.random() < campaign['response_rate']
                
                campaigns.append({
                    'customer_id': customer['customer_id'],
                    'campaign_name': campaign['name'],
                    'campaign_start': campaign['start'],
                    'campaign_end': campaign['end'],
                    'responded': responded,
                    'channel': np.random.choice(['Email', 'SMS', 'Push', 'Social'])
                })
        
        return pd.DataFrame(campaigns)


def main():
    """Generate and save the complete dataset."""
    
    print("Starting data generation...")
    generator = EcommerceDataGenerator(seed=42)
    
    # Generate datasets
    customers_df, transactions_df = generator.generate_complete_dataset(n_customers=10000)
    campaigns_df = generator.add_marketing_campaigns(customers_df, transactions_df)
    
    # Save to files
    print("Saving datasets...")
    customers_df.to_csv('data/raw/customers.csv', index=False)
    transactions_df.to_csv('data/raw/transactions.csv', index=False)
    campaigns_df.to_csv('data/raw/campaigns.csv', index=False)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Customers: {len(customers_df):,}")
    print(f"Transactions: {len(transactions_df):,}")
    print(f"Campaign interactions: {len(campaigns_df):,}")
    print(f"Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
    print(f"Total revenue: ${transactions_df['final_price'].sum():,.2f}")
    
    # Segment distribution
    print("\nCustomer Segment Distribution:")
    print(customers_df['segment'].value_counts(normalize=True).round(3))
    
    print("\nData generation complete!")


if __name__ == "__main__":
    main()