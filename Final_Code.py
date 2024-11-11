import pandas as pd
import numpy as np
import time
import random
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, PULP_CBC_CMD

# Load the data
nutriets = pd.read_csv("/content/nutrient-categories.csv")
constraints = pd.read_csv("/content/nutrient-constraints.csv")[['Nutrient Name', 'Lower Bound', 'Upper Bound', 'Unit']]

# Filter the nutrient names to match both datasets
keep_names = list(set(nutriets.columns).intersection(set(constraints['Nutrient Name'])))

# Select relevant columns
nutriets = nutriets[['Category'] + keep_names]
constraints = constraints[constraints['Nutrient Name'].isin(keep_names)]

# Data cleansing
nutriets = nutriets.apply(lambda x: x / 100 if x.name != 'Category' else x)

# Set an upper caloric limit of 2500 kcal
constraints.loc[constraints['Nutrient Name'] == 'Energ_Kcal', 'Upper Bound'] = 2500
constraints['Upper Bound'] = np.where(constraints['Upper Bound'] < constraints['Lower Bound'], constraints['Lower Bound'] * 2, constraints['Upper Bound'])

# Number of days to plan
all_days = 7
foods_used = ['WATER']  # Keep water always in the plan
all_results = []

for day in range(1, all_days + 1):
    # Sample the data (here we sample 1, change it as needed)
    sample_nutriets = nutriets.sample(frac=0.3)  # Sample 50% of data for debugging

    # Remove already used foods
    sample_nutriets = sample_nutriets[~sample_nutriets['Category'].isin(foods_used)]
    print(f"DAY {day}: Foods remaining for optimization: {sample_nutriets.shape[0]} foods")

    # Check if sample_nutrients is empty before proceeding
    if sample_nutriets.empty:
        print(f"DAY {day}: No food options available for optimization.")
        continue  # Skip to the next day

    # Set objective: minimize carbohydrates
    objective_function = sample_nutriets['Carbohydrt_(g)'].values.flatten()  # Ensure it's a 1D array

    # Initialize the LP problem
    prob = LpProblem(f"DietOptimization_Day_{day}", LpMinimize)

    # Create a dictionary of LP variables for the food items (non-negative quantities)
    food_vars = {food: LpVariable(food, lowBound=0, cat='Continuous') for food in sample_nutriets['Category']}

    # Objective function: minimize the carbohydrates
    prob += sum(objective_function[i] * food_vars[food] for i, food in enumerate(sample_nutriets['Category'])), "Total_Carb_Intake"

    # Add constraints
    for _, row in constraints.iterrows():
        constraint_name = row['Nutrient Name']
        lower_bound = row['Lower Bound']
        upper_bound = row['Upper Bound']

        # Get the nutrient column
        nutrient_column = sample_nutriets[constraint_name].values

        # Add lower bound constraint
        prob += sum(nutrient_column[i] * food_vars[food] for i, food in enumerate(sample_nutriets['Category'])) >= lower_bound, f"{constraint_name}_lower"

        # Add upper bound constraint
        prob += sum(nutrient_column[i] * food_vars[food] for i, food in enumerate(sample_nutriets['Category'])) <= upper_bound, f"{constraint_name}_upper"

    # Solve the problem using the CBC solver with debugging logs
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(timeLimit=10000))  # Using the CBC solver with logs

    elapsed_time = time.time() - start_time

    # Check if the problem was solved successfully
    if LpStatus[prob.status] == 'Optimal':
        selected_foods = [food for food in sample_nutriets['Category'] if food_vars[food].varValue > 0]
        amounts = [food_vars[food].varValue for food in selected_foods]

        # Store the results for the day
        day_results = pd.DataFrame({'Food': selected_foods, 'Amount(g)': amounts, 'Day': day})
        all_results.append(day_results)

        # Add the selected foods to the used list
        foods_used.extend(selected_foods)

        print(f"DAY {day}: {len(selected_foods)} items selected. LP completed in {elapsed_time:.2f} seconds.")
    else:
        # Provide feedback on solver status
        print(f"DAY {day}: LP solver failed with status {LpStatus[prob.status]}.")

# Function to print results in a formatted table
def print_results(all_results):
    """
    Prints the daily diet results in a formatted table.

    Args:
        all_results (list): List of DataFrames containing daily diet results.
    """

    # Find the maximum number of rows across all DataFrames
    max_rows = max(len(df) for df in all_results)

    # Create a DataFrame with empty rows for alignment
    blank_rows = pd.DataFrame({"rownum": range(1, max_rows + 1)})

    # Process each DataFrame in the list
    all_results_processed = []
    for results_df in all_results:
        # Add a "rownum" column for merging
        results_df["rownum"] = range(1, len(results_df) + 1)

        # Merge with blank rows to ensure consistent length
        merged_df = blank_rows.merge(results_df, on="rownum", how="left").drop("rownum", axis=1)

        # Rename columns based on day and food/amount
        day = int(results_df["Day"].iloc[0])
        merged_df.columns = [f"Day {day} food", f"Day {day} amt(g)", "Day"]

        # Append the processed DataFrame
        all_results_processed.append(merged_df)

    # Concatenate all processed DataFrames
    all_results_print = pd.concat(all_results_processed, axis=1)

    # Print the final table using pandas.to_string() for better readability
    print(all_results_print.to_string(index=False))

# Print the results in a formatted table
print_results(all_results)

# Save the results to a CSV file
if all_results:
    final_results = pd.concat(all_results)
    final_results.to_csv("optimized_food_plan_pulp.csv", index=False)
    print("Optimization completed. Results saved in 'optimized_food_plan_pulp.csv'.")
else:
    print("No valid results to save.")

# Sensitivity Analysis Function
def sensitivity_analysis(constraints, prob, food_vars, sample_nutriets, adjustment=0.1):
    """
    Conducts sensitivity analysis by adjusting nutrient upper bounds and observing changes.

    Args:
        constraints (DataFrame): Nutrient constraints with bounds.
        prob (LpProblem): Optimization problem.
        food_vars (dict): Dictionary of LP food variables.
        sample_nutriets (DataFrame): Nutrients data for selected food.
        adjustment (float): Adjustment percentage for sensitivity analysis.
    """
    print("\nSensitivity Analysis Results:")
    for index, row in constraints.iterrows():
        nutrient = row['Nutrient Name']
        original_upper = row['Upper Bound']

        # Increase and decrease the upper bound slightly
        new_upper = original_upper * (1 + adjustment)
        constraints.loc[index, 'Upper Bound'] = new_upper

        # Re-run optimization with new constraint
        prob_copy = prob.copy()
        prob_copy.constraints[f"{nutrient}_upper"] = sum(sample_nutriets[nutrient].values[i] * food_vars[food] for i, food in enumerate(sample_nutriets['Category'])) <= new_upper
        prob_copy.solve(PULP_CBC_CMD(msg=False))

        if LpStatus[prob_copy.status] == 'Optimal':
            total_carbs = sum(food_vars[food].varValue * sample_nutriets.loc[sample_nutriets['Category'] == food, 'Carbohydrt_(g)'].values[0] for food in food_vars if food_vars[food].varValue > 0)
            print(f"{nutrient} upper bound changed to {new_upper:.2f}. New total carbs: {total_carbs:.2f}")
        else:
            print(f"{nutrient} adjustment led to infeasible solution.")

        # Reset the original upper bound
        constraints.loc[index, 'Upper Bound'] = original_upper

# Example usage of sensitivity analysis
sensitivity_analysis(constraints, prob, food_vars, sample_nutriets, adjustment=0.1)