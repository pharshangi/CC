# CC
CC_Part1


# 1. How did you verify that you are parsing the contours directly?
# A : I manually checked if the code was working for one mask by plotting it on the image file.
# 
# 2. Any changes made to the given code 'parsing.py'?
# A : No.



# 1. I made a separate one for part 2. Definitely could make it more efficient in resouces and space. By randomizing the patient id and also the contour file that was selected.
# 
# 2. Same as before : Manually checking the masking. However, we could make it more robust by keeping a count of (sum of all pixels - sum of all pixels in contour) per image pair.
# 
# 3. Make it a function call rather than stand alone part of the code. Would make the code much more easier to read. Especially the second part of the loop.
