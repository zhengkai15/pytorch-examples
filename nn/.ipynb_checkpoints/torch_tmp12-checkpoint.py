import torch
import matplotlib.pyplot as plt

w = 2
b = 1
noise = torch.rand(100, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# 因为输入层格式要为(-1, 1)，所以这里将(100)的格式转成(100, 1)
y = w*x + b + noise
# 拟合分布在y=2x+1上并且带有噪声的散点
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
    )
# 自定义的网络，带有2个全连接层和一个tanh层
loss_fun = torch.nn.MSELoss()
# 定义损失函数为均方差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 使用adam作为优化器更新网络模型的权重，学习率为0.01

plt.ion()
# 图形交互
for _ in range(10000):
    ax = plt.axes()
    output = model(x)
    # 数据向后传播（经过网络层的一次计算）
    loss = loss_fun(output, y)
    # 计算损失值
    # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
    # print("-"*100)
    model.zero_grad()
    # 优化器清空梯度
    # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
    # print("-"*100)
    # 通过注释地方可以对比发现执行zero_grad方法以后倒数梯度将会被清0
    # 如果不清空梯度的话，则会不断累加梯度，从而影响到当前梯度的计算
    loss.backward()
    # 向后传播，计算当前梯度，如果这步不执行，那么优化器更新时则会找不到梯度
    optimizer.step()
    # 优化器更新梯度参数，如果这步不执行，那么因为梯度没有发生改变，loss会一直计算最开始的那个梯度
    if _ % 100 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        # print("w:", list(model.children())[0].weight.t() @ list(model.children())[-1].weight.t())
        # 通过这句可以查看权值变化，可以发现最后收敛到2附近

plt.ioff()
plt.show()