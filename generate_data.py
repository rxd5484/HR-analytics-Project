"""
Generate Realistic HR Employee Dataset
Creates 15,000 employee records with realistic patterns and correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_employee_data(n_employees=15000):
    """Generate synthetic but realistic employee dataset"""
    
    data = []
    
    departments = ['Sales', 'IT', 'HR', 'Operations', 'Finance']
    job_roles = {
        'Sales': ['Sales Executive', 'Sales Manager', 'Account Manager'],
        'IT': ['Software Engineer', 'Data Scientist', 'DevOps Engineer', 'Product Manager'],
        'HR': ['HR Manager', 'Recruiter', 'HR Business Partner'],
        'Operations': ['Operations Manager', 'Logistics Coordinator', 'Supply Chain Analyst'],
        'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager']
    }
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    marital_status = ['Single', 'Married', 'Divorced']
    
    for emp_id in range(1, n_employees + 1):
        # Demographics
        age = np.random.randint(22, 60)
        gender = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        education = np.random.choice(education_levels, p=[0.15, 0.50, 0.30, 0.05])
        marital = np.random.choice(marital_status, p=[0.35, 0.50, 0.15])
        
        # Job characteristics
        department = np.random.choice(departments)
        job_role = np.random.choice(job_roles[department])
        years_at_company = max(0, min(age - 22, np.random.gamma(3, 2)))
        years_in_role = min(years_at_company, np.random.gamma(2, 1.5))
        
        # Performance & satisfaction
        job_satisfaction = np.random.randint(1, 5)
        environment_satisfaction = np.random.randint(1, 5)
        relationship_satisfaction = np.random.randint(1, 5)
        work_life_balance = np.random.randint(1, 5)
        
        performance_rating = np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.60, 0.20])
        
        # Compensation
        education_multiplier = {'High School': 1.0, 'Bachelor': 1.3, 'Master': 1.6, 'PhD': 2.0}
        base_salary = {
            'Sales': 4500, 'IT': 6000, 'HR': 4000,
            'Operations': 4200, 'Finance': 5500
        }
        
        monthly_income = base_salary[department] * education_multiplier[education]
        monthly_income *= (1 + years_at_company * 0.03)  # 3% per year
        monthly_income *= (1 + (performance_rating - 2) * 0.1)
        monthly_income = int(monthly_income + np.random.normal(0, 500))
        
        # Work characteristics
        overtime = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        distance_from_home = np.random.gamma(2, 5)
        num_companies_worked = min(int(np.random.gamma(2, 1)), int(years_at_company / 2) + 1)
        years_since_last_promotion = min(years_at_company, int(np.random.exponential(2)))
        training_times_last_year = np.random.randint(0, 7)
        
        # Calculate attrition based on realistic factors
        attrition_score = 0
        
        # Negative factors (increase attrition)
        if job_satisfaction <= 2:
            attrition_score += 0.35
        if work_life_balance <= 2:
            attrition_score += 0.25
        if monthly_income < base_salary[department] * 1.2:
            attrition_score += 0.20
        if overtime == 'Yes':
            attrition_score += 0.15
        if years_at_company < 2:
            attrition_score += 0.25
        if years_since_last_promotion > 4:
            attrition_score += 0.20
        if distance_from_home > 20:
            attrition_score += 0.10
        if environment_satisfaction <= 2:
            attrition_score += 0.15
        
        # Positive factors (decrease attrition)
        if job_satisfaction >= 4:
            attrition_score -= 0.30
        if work_life_balance >= 4:
            attrition_score -= 0.20
        if performance_rating == 4:
            attrition_score -= 0.15
        if training_times_last_year >= 4:
            attrition_score -= 0.10
        if years_at_company >= 5:
            attrition_score -= 0.20
        
        # Department-specific attrition rates
        dept_attrition = {'Sales': 0.28, 'IT': 0.12, 'HR': 0.15, 'Operations': 0.18, 'Finance': 0.14}
        attrition_score += (dept_attrition[department] - 0.16)
        
        # Final attrition decision
        attrition_prob = 1 / (1 + np.exp(-attrition_score * 5))  # Sigmoid
        attrition = 1 if np.random.random() < attrition_prob else 0
        
        employee = {
            'EmployeeID': emp_id,
            'Age': int(age),
            'Gender': gender,
            'MaritalStatus': marital,
            'Education': education,
            'Department': department,
            'JobRole': job_role,
            'YearsAtCompany': round(years_at_company, 1),
            'YearsInCurrentRole': round(years_in_role, 1),
            'YearsSinceLastPromotion': int(years_since_last_promotion),
            'MonthlyIncome': int(monthly_income),
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'RelationshipSatisfaction': relationship_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'PerformanceRating': performance_rating,
            'OverTime': overtime,
            'DistanceFromHome': round(distance_from_home, 1),
            'NumCompaniesWorked': num_companies_worked,
            'TrainingTimesLastYear': training_times_last_year,
            'Attrition': attrition
        }
        
        data.append(employee)
    
    df = pd.DataFrame(data)
    
    # Add some calculated features
    df['TenureToIncomeRatio'] = df['YearsAtCompany'] / (df['MonthlyIncome'] / 1000)
    df['SatisfactionScore'] = (
        df['JobSatisfaction'] + 
        df['EnvironmentSatisfaction'] + 
        df['RelationshipSatisfaction'] + 
        df['WorkLifeBalance']
    ) / 4
    
    print(f"Dataset created: {len(df)} employees")
    print(f"Attrition rate: {df['Attrition'].mean()*100:.1f}%")
    print(f"\nAttrition by department:")
    print(df.groupby('Department')['Attrition'].mean().sort_values(ascending=False) * 100)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_employee_data(15000)
    
    # Save to CSV
    import os
    os.makedirs('../data', exist_ok=True)
    

    df.to_csv('data/employee_data.csv', index=False)
    print(f"\nDataset saved to data/employee_data.csv")
    
    # Print summary statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nColumn types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())