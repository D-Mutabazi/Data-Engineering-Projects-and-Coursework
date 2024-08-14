import torch

scalar = torch.tensor(7) 
print(scalar.ndim)
scalar.item() #covert type to python integer - Get the Python number within a tensor 
scalar = scalar.item()
print(scalar)

random_tensor = torch.rand(size =(3,4)) #create a random tensor of zise 3,4
# print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size = (224, 224, 3)) #create a random image size tensor
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

################# ----- Zero's and Ones ############################

zeros = torch.zeros(size =(3,4)) # Create a tensor of all zeros
print(zeros, zeros.dtype)

ones = torch.ones(size = (3,4)) # Create a tensor of all ones
print(ones, ones.dtype)

############### --- Create a range and tensors like --- ##########

zero_to_ten = torch.arange(start=0, end=10, step=1) # Create a range of values 0 to 10
print(zero_to_ten)
