import numpy as np

def find_lcm():
    # Get three numbers from the user
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    num3 = int(input("Enter the third number: "))

    # Calculate the Least Common Multiple (LCM)
    lcm = np.lcm.reduce([num1, num2, num3])
    print(f"The Least Common Multiple of {num1}, {num2}, and {num3} is {lcm}")

# Call the function
find_lcm()
