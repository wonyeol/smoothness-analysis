x = pyro.sample("sampx", Normal(0, 1))
a = x
b = 0
c = x + 3 
while (a < c):
    a = a - 3
    b = b + 1 
