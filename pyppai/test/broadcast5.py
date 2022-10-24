mu = torch.zeros([2, 1, 4, 5])
scale = torch.rand([1, 4, 1])

x = pyro.sample("x", Normal(mu,scale), obs=torch.randn([2, 1, 4, 5]))
