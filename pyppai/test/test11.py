x = pyro.sample( "sampx", Normal(0, 1))
if x < 2: 
    x = 8
