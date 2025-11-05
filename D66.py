'''Data Aggregation
Problem Statement: Analyzing Sales Performance by Region in a Retail Company'''


import pandas as pd

df = pd.read_csv(r"D:\BE_Practicals\DMVcodes\DMV_Practical\6_Retail_Sales.csv")

print(df.info())

print(df)

sales_by_region = df.groupby('Region')['Sales_Amount'].sum().reset_index()
print(sales_by_region)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.pie(sales_by_region['Sales_Amount'], labels=sales_by_region['Region'], autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Region')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

top_performing_regions = sales_by_region.sort_values(by='Sales_Amount', ascending=False)

# Print the top-performing regions
print("Top-performing regions based on highest sales amount:")
print(top_performing_regions)

sales_by_region_category = df.groupby(['Region', 'Product_category'])['Sales_Amount'].sum().reset_index()

# Display the result
print(sales_by_region_category)

ax = sales_by_region_category.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.title('Sales Comparison by Region and Product Category')
plt.legend(title='Product Category', loc='upper right', bbox_to_anchor=(1.15, 1))

# Display the plot
plt.show()
