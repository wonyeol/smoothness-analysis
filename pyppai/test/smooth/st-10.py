# Smoothness analysis test
# Basic coefficients

x = pyro.sample( "x", Uniform(1., 4.))
y = pyro.sample( "y", Uniform(-1., 3.))
z = x + y
t = x * y
u = 2 * x
