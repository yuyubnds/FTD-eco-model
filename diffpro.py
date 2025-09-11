import numpy as np
import matplotlib.pyplot as plt

# 设置参数
L = 10.0        # 空间长度
T = 20        # 时间长度
alpha = 0.01    # 热扩散系数
Nx = 8        # 空间离散点数
Ny = 8
Nt = 50     # 时间离散点数
dx = L / (Nx)  # 空间步长
dy = L / (Ny)  # 空间步长
dt = T / Nt        # 时间步长

# 初始化温度分布
x = np.linspace(0, L, Nx)
u = np.sin(np.pi * x / L)  # 初始条件：u(x, 0) = sin(pi * x / L)

# 结果存储
y1 = np.zeros((10,10), dtype=np.float64)
y2 = np.zeros((10,10), dtype=np.float64)
y3 = np.zeros((10,10), dtype=np.float64)
y1_new = np.zeros((10,10), dtype=np.float64)
y2_new = np.zeros((10,10), dtype=np.float64)
y3_new = np.zeros((10,10), dtype=np.float64)

for i in range(0,10):
    for j in range(0,10):
        y1[i,j]=5000
        y2[i,j]=100000
        y3[i,j]=10000

t = np.zeros(50, dtype=np.float64)

# 时间迭代
for n in range(1, Nt+1):
    y1_new = y1.copy()
    y2_new = y2.copy()
    y3_new = y3.copy()
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            y1_new[i,j] = y1[i,j] + dt / (dx**2) * 0.0001 * (0.002*(y2[i+1,j] - 2*y2[i,j] + y2[i-1,j])) + dt / (dy**2) * 0.0001 * (0.002*(y2[i,j+1] - 2*y2[i,j] + y2[i,j-1])) + dt*((0.4+0.05*np.sin(n*dt))*y1[i,j]*(1-y1[i,j]/10000) - 0.03*y1[i,j] - 0.008/100000*y1[i,j]*y2[i,j] + 0.04*y3[i,j])
            y2_new[i,j] = y2[i,j] + dt / (dx**2) * 0.3 * (0.03*(y1[i+1,j] - 2*y1[i,j] + y1[i-1,j]) - 0.15*(y3[i+1,j] - 2*y3[i,j] + y3[i-1,j])) + dt / (dy**2) * 0.3 * (0.03*(y1[i,j+1] - 2*y1[i,j] + y1[i,j-1]) - 0.15*(y3[i,j+1] - 2*y3[i,j] + y3[i,j-1])) + dt * ((0.5+0.055*np.sin(n*dt))*y2[i,j]*(1-y2[i,j]/(950000+(n*dt*10000/2/3.14)%10000)) - 0.0015/1000*y2[i,j]*y3[i,j] - 0.3*y2[i,j] + 0.00008*y1[i,j])
            y3_new[i,j] = y3[i,j] + dt / (dx**2) * 0.8 * (0.01*(y2[i+1,j] - 2*y2[i,j] + y2[i-1,j])) + dt / (dy**2) * 0.8 * (0.01*(y2[i,j+1] - 2*y2[i,j] + y2[i,j-1])) + dt * ((0.25+0.02*np.sin(n*dt))*y3[i,j]*(1-y3[i,j]/50000) - 0.01*y3[i,j] + 0.001*y2[i,j])
    y1=y1_new
    y2=y2_new
    y3=y3_new

# 可视化结果
print(y1)
print(y2)
print(y3)

plt.imshow(y1, origin='lower', cmap='magma')
plt.colorbar(label='Temperature')
plt.title('Temperature Distribution Heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
