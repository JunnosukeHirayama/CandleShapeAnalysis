import numpy as np
import matplotlib.pyplot as plt

# 1. パラメータ設定 

Nr = 40
Nz = 80
R_max = 0.04     # 4cm
Z_max = 0.12     # 12cm

# 芯の半径 (1.5mm)
R_wick = 0.0015

# 物理定数
rho0 = 1.0
nu = 0.0001      # 動粘性係数
D_Z = 0.0005     # 拡散係数 
g = 9.81
beta = 1/300.0

# 温度・燃料パラメータ
T0 = 300.0       # 室温 (約27℃)
T_fuel = 600.0   # 芯の温度

T_f = 1500.0     

Z_st = 0.055

#
# 流速設定 (弱火キープ)
V_in_target = 0.03

# 時間設定
dt = 0.0001
n_steps = 15000

# グリッド生成
dr = R_max / Nr
dz = Z_max / Nz
r = np.linspace(dr/2, R_max-dr/2, Nr)
z = np.linspace(dz/2, Z_max-dz/2, Nz)
R, Z_grid = np.meshgrid(r, z)

# 2. 変数初期化
u = np.zeros((Nz, Nr))
v = np.zeros((Nz, Nr))
Z = np.zeros((Nz, Nr))

# 3. 境界条件関数

def apply_boundary_conditions(u, v, Z, current_step, total_ramp_steps=1000):
    factor = min(1.0, current_step / total_ramp_steps)
    current_Vin = V_in_target * factor

    # 芯
    wick_mask = r <= R_wick
    v[0, wick_mask] = current_Vin
    u[0, wick_mask] = 0.0
    Z[0, wick_mask] = 1.0

    # 壁
    wall_mask = r > R_wick
    v[0, wall_mask] = 0.0
    u[0, wall_mask] = 0.0
    Z[0, wall_mask] = 0.0

    # 側面
    u[:, -1] = 0.0
    Z[:, -1] = 0.0
    
    # 中心軸
    u[:, 0] = 0.0
    
    # 天井
    v[-1, :] = v[-2, :]
    u[-1, :] = u[-2, :]
    Z[-1, :] = Z[-2, :]

    return u, v, Z

# 4. ヘルパー関数

def shift_up(f):
    res = np.zeros_like(f)
    res[:-1, :] = f[1:, :]
    res[-1, :] = f[-1, :]
    return res

def shift_down(f):
    res = np.zeros_like(f)
    res[1:, :] = f[:-1, :]
    res[0, :] = f[0, :]
    return res

def shift_left(f):
    res = np.zeros_like(f)
    res[:, :-1] = f[:, 1:]
    res[:, -1] = f[:, -1]
    return res

def shift_right(f):
    res = np.zeros_like(f)
    res[:, 1:] = f[:, :-1]
    res[:, 0] = f[:, 0]
    return res

# 5. メインループ

print(f"Simulation Start: {n_steps} steps (Temp 1500K)")
np.seterr(all='ignore')

r_safe = r + 1e-10

for n in range(n_steps):
    
    # 温度計算
    T = np.full_like(Z, T0)
    mask1 = Z < Z_st
    T[mask1] = T0 + (T_f - T0) * (Z[mask1] / Z_st)
    mask2 = Z >= Z_st
    T[mask2] = T_f - (T_f - T_fuel) * ((Z[mask2] - Z_st) / (1.0 - Z_st))

    buoyancy = g * beta * (T - T0)

    # v 
    dv_dz = (v - shift_down(v)) / dz
    dv_dr = (shift_left(v) - shift_right(v)) / (2*dr)
    advection_v = v * dv_dz + u * dv_dr
    
    d2v_dz2 = (shift_up(v) - 2*v + shift_down(v)) / dz**2
    d2v_dr2 = (shift_left(v) - 2*v + shift_right(v)) / dr**2
    diffusion_v = nu * (d2v_dz2 + d2v_dr2 + (1.0/r_safe)*dv_dr)
    
    v_new = v + dt * (diffusion_v - advection_v + buoyancy)

    # u 
    du_dz = (u - shift_down(u)) / dz
    du_dr = (shift_left(u) - shift_right(u)) / (2*dr)
    advection_u = v * du_dz + u * du_dr
    
    d2u_dz2 = (shift_up(u) - 2*u + shift_down(u)) / dz**2
    d2u_dr2 = (shift_left(u) - 2*u + shift_right(u)) / dr**2
    diffusion_u = nu * (d2u_dz2 + d2u_dr2 + (1.0/r_safe)*du_dr - u/(r_safe**2))
    
    u_new = u + dt * (diffusion_u - advection_u)

    # Z
    dZ_dz = (Z - shift_down(Z)) / dz
    dZ_dr = (shift_left(Z) - shift_right(Z)) / (2*dr)
    advection_Z = v * dZ_dz + u * dZ_dr
    
    d2Z_dz2 = (shift_up(Z) - 2*Z + shift_down(Z)) / dz**2
    d2Z_dr2 = (shift_left(Z) - 2*Z + shift_right(Z)) / dr**2
    diffusion_Z = D_Z * (d2Z_dz2 + d2Z_dr2 + (1.0/r_safe)*dZ_dr)
    
    Z_new = Z + dt * (diffusion_Z - advection_Z)

    # 更新
    u = np.clip(u_new, -1.0, 1.0)
    v = np.clip(v_new, -0.5, 4.0)
    Z = np.clip(Z_new, 0.0, 1.0)
    
    u, v, Z = apply_boundary_conditions(u, v, Z, n)

    if n % 3000 == 0:
        print(f"Step {n}/{n_steps} done. Max Temp: {np.max(T):.1f} K")

# 6. 可視化

print("Plotting results...")
plt.figure(figsize=(10, 8))


# 1. 左右反転させたものを用意
T_left = np.fliplr(T)
Z_left = np.fliplr(Z)
R_left = -np.fliplr(R)

# 2. 中心(r=0)のデータを補完
T_center = T[:, 0:1]
Z_center = Z[:, 0:1]
R_center = np.zeros((Nz, 1)) 

# 3. [左] + [中心] + [右] に結合
T_combined = np.hstack([T_left, T_center, T])
Z_data_combined = np.hstack([Z_left, Z_center, Z]) 
R_combined = np.hstack([R_left, R_center, R])

# 縦軸座標もサイズを合わせるために結合
Z_grid_combined = np.hstack([Z_grid, Z_grid[:, 0:1], Z_grid])


# 左図：温度分布
plt.subplot(1, 2, 1)

# 描画
cf = plt.contourf(R_combined*100, Z_grid_combined*100, T_combined, 
                  levels=100, cmap='inferno')

# 炎の輪郭線
try:
    if np.max(Z_data_combined) >= Z_st:
        plt.contour(R_combined*100, Z_grid_combined*100, Z_data_combined, 
                    levels=[Z_st], colors='cyan', linewidths=2)
except:
    pass

plt.colorbar(cf, label='Temperature [K]', shrink=0.8)
plt.title('Teardrop Flame Shape', fontsize=14)
plt.xlabel('r [cm]')
plt.ylabel('z [cm]')
plt.xlim(-R_max*100, R_max*100)
plt.ylim(0, Z_max*100)
plt.gca().set_aspect('equal')

# 右図：流速場
plt.subplot(1, 2, 2)

step_r = 2
step_z = 2
plt.quiver(R[::step_z, ::step_r]*100, Z_grid[::step_z, ::step_r]*100, 
           u[::step_z, ::step_r], v[::step_z, ::step_r], 
           scale=3, color='black', alpha=0.5)

plt.title('Velocity Field', fontsize=14)
plt.xlabel('r [cm]')
plt.ylabel('z [cm]')
plt.xlim(0, R_max*100)
plt.ylim(0, Z_max*100)
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show(block=True)