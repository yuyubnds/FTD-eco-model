import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义微分方程组
def diff_eq(t, y):
    y1, y2, y3, yd = y
    dy1_dt = (0.4+0.05*np.sin(t))*y1*(1-y1/10000) - 0.03*y1 - 0.008/100000*y1*y2 + 0.04*y3
    dy2_dt = (0.5+0.055*np.sin(t))*y2*(1-y2/(950000+(t*10000/2/3.14)%10000)) - 0.0015/1000*y2*y3 - 0.3*y2 + 0.00008*y1
    dy3_dt = (0.25+0.02*np.sin(t))*y3*(1-y3/50000) - 0.1*y3 + 0.0011*y2
    dyd_dt = 0.05*yd*(1-yd/500000) + 0.2/10*(0.03*y1 + 0.125*y2 + 0.125*y3) - 0.04*yd
    return np.array([dy1_dt, dy2_dt, dy3_dt, dyd_dt], dtype=np.float64)

# 初始条件
y0 = np.array([5000, 100000, 10000, 750000], dtype=np.float64)

# 时间范围
t_span = (0, 50)
t_eval = np.linspace(0, 50, 100, dtype=np.float64)

# 求解微分方程组
solution = solve_ivp(diff_eq, t_span, y0, t_eval=t_eval)

# 提取解
t = solution.t
y1 = solution.y[0]
y2 = solution.y[1]
y3 = solution.y[2]
yd = solution.y[3]

# 绘图
plt.plot(t, y1, label="y1(t)")
plt.plot(t, y2, label="y2(t)")
plt.plot(t, y3, label="y3(t)")
plt.plot(t, yd, label="yd(t)")
plt.xlabel("Time (t)")
plt.ylabel("Solution (y)")
plt.title("Solution of the Differential Equation System")
plt.legend()
plt.grid()
plt.show()

print(t)
print(y1)
print(y2)
print(y3)
print(yd)

y=y1+y2+y3+yd
p1=y1/y
p2=y2/y
p3=y3/y
pd=yd/y
H=p1*np.log(p1)+p2*np.log(p2)+p3*np.log(p3)+pd*np.log(pd)
H=-H
plt.plot(t, H, label="H")
plt.xlabel("Time (t)")
plt.ylabel("Solution (H)")
plt.title("Solution of the Differential Equation System")
plt.legend()
plt.grid()
plt.show()
print(H)
