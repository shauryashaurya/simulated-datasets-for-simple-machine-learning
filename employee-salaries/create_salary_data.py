import pandas as pd
import numpy as np

# How much data do you need?
num_rows = 15000

# Base salary increase per year of experience
base_salary = 36000
increase_per_year = 1200 

# Define parameters
years_exp_range = (1, 20)  
education_levels = ['Bachelor', 'Master', 'PhD', 'MBA']
job_categories = ['Software Developer', 'Data Analyst', 'Data Scientist', 'Business Analyst', 'QA Engineer', 'Project Manager']
company_sizes = ['Small', 'Medium', 'Large']

# Generate sample data
np.random.seed(42)  
data = {
    'years_experience': np.random.randint(years_exp_range[0], years_exp_range[1] + 1, size=num_rows),
    'education_level': np.random.choice(education_levels, size=num_rows),
    'job_category': np.random.choice(job_categories, size=num_rows),
    'company_size': np.random.choice(company_sizes, size=num_rows) 
}

# Simulate salaries 
df = pd.DataFrame(data)
df['salary'] = base_salary + df['years_experience'] * increase_per_year + np.random.normal(0, 5000, size=num_rows)

# Simulate salaries with correlation to education level
base_adjustment = {'Bachelor': 0, 'Master': 5000, 'PhD': 12000, 'MBA':20000}
df['salary'] = base_salary + df['years_experience'] * increase_per_year + df['education_level'].map(base_adjustment)

# Add noise to education-based salary adjustment
noise_factor = 5000  # Control the magnitude of noise
df['salary'] += np.random.normal(0, noise_factor, size=num_rows)

# Simulate number of employees with correlation to company size
df.loc[df['company_size'] == 'Small', 'num_employees'] = np.random.randint(50, 500, size=df['company_size'].value_counts()['Small'])
df.loc[df['company_size'] == 'Medium', 'num_employees'] = np.random.randint(500, 2000, size=df['company_size'].value_counts()['Medium'])
df.loc[df['company_size'] == 'Large', 'num_employees'] = np.random.randint(2000, 10000, size=df['company_size'].value_counts()['Large'])

# Rounding Modifications
df['salary'] = df['salary'].round().astype(int)  # Round salary to integer
df['num_employees'] = df['num_employees'].astype(int)  # Cast num_employees to integer

# Exporting Data
df.to_csv('simulated_salary_data.csv', index=False) 
