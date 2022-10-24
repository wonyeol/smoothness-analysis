# Smoothness analysis test
# condition with path known due to sampling

x = pyro.sample( "x", Uniform(1., 4.))
if x < 0:
    y = 0
    t = a
else:
    y = x
    t = b

z = y * x * t
