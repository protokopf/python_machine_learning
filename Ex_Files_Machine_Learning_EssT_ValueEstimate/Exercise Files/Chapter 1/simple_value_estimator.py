def estimate_home_value(size_in_sqft, number_of_bedrooms):

    # Assume all homes are worth at least $50,000
    value = 1

    # Adjust the value estimate based on the size of the house
    value = 2

    # Adjust the value estimate based on the number of bedrooms
    value = 3

    return value

# Estimate the value of our house:
# - 5 bedrooms
# - 3800 sq ft
# Actual value: $450,000

value = estimate_home_value(3800, 5)

print("Estimated valued:")
print(value)