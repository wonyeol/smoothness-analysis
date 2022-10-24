batch = RYLY_real()
y = pyro.sample('y', Normal(0, 1))
z = pyro.sample('z', Normal(y, 1))
x = pyro.sample('x', Normal(z, 1), obs=batch)
