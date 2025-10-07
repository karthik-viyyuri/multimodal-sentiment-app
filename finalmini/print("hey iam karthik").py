print("hey iam karthik")
print(" i like cosing and editing")
 #variable : it is a container for a value, it stores a value and acts as it.
first_name = 'Karthik'

print(first_name)

last_name = 'Viyyuri'

print(f"hello mr.{last_name}")

#strings : they are the words and letters using alphabets.
food = "pasta" 
email = "karthikviyyuri@gmail.com"
print(f"would you like to order a {food} and i can send the bill to {email}")

#integers : they are numbers 
age = 21
print(f"you are {age} years old")

#float : it is a decimal number.
gpa = 7.68
print(f"your GPA is {gpa}")

#Boolean : true or fasle
is_student = False
print(f"are you a student? {is_student}")

if is_student:
    print("you are a student")
else:
    print("you are not a student")
    
#Type Casting : the process of converting one datatype to another datatype.
print(type(first_name))
print(type(gpa))

gpa = int(gpa)
gpa += 1 
print(gpa)

#input

name = int(input("what is your name? "))
print(f"hello {name}")

