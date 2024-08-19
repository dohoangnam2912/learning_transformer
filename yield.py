def count_up_to(max_value):
    count = 1
    while count <= max_value:
        # yield count
        print(f'Count: {count}')
        count += 1

# Create a generator object
counter = count_up_to(5)

# Iterate over the generator
for number in counter:
    print(number)
