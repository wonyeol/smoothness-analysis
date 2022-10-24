#
#
x = pyro.param("x", torch.ones([1,3,1]))
y = pyro.param("y", torch.ones([3,2]))
z = pyro.param("z", torch.ones([5,4,3,1])) 
v = x + z
w = x + y
a = torch.ones([4,1]) + torch.ones([4])
