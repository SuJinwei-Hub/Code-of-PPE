import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit    # 加载拟合模块
from scipy.interpolate import make_interp_spline


standard_position = 2       # 校准位置：2mm
c2 = 1.82384
gamma1 = 0.81799
gamma2 = 0.29115
kbt = 4.14
lp1 = []
l01 = []
lp2 = []
l02 = []
k0 = []
uplim0 = 500            # 拟合参数0上界
uplim1 = 5000           # 拟合参数1上界
uplim2 = 5000           # 拟合参数2上界


def func1(x, lp, l0):
    return kbt/lp*(0.25/(1-x/l0)**2-0.25+x/l0)


def func2(x, lp, l0, k0):
    return l0*(1-0.5*(kbt/x/lp)**(0.5)+x/k0)


def text_save(filename, data):      # filename为写入txt文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')    # 去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'     # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()


def get_data(txtname):
    table = pd.read_table(txtname)
    table = np.array(table)
    t1 = table[:, 1]
    t2 = table[:, 3]
    t1 = np.delete(t1, 0, axis=0)
    t2 = np.delete(t2, 0, axis=0)
    return t1, t2


def filter_nan(magd, d):
    de = []
    for i in range(0, len(d)):
        if pd.isna(d[i]):     # 寻找元素为nan的索引
            de.append(i)
    magd_new = np.delete(magd, de, axis=0)
    d_new = np.delete(d, de, axis=0)
    return magd_new, d_new


def magd2f(magd, force):
    c1 = force/(np.exp(-standard_position/gamma1)+c2*np.exp(-standard_position/gamma2))
    f = c1*(np.exp(-(magd/gamma1))+c2*np.exp(-(magd/gamma2)))
    return f


def segment(d, f):
    index = np.where(f == np.max(f))
    max_index = index[0][0]            # 最小值的序列是一段，取序列第一个为索引
    d1 = d[0:max_index]
    f1 = f[0:max_index]
    d2 = d[max_index:len(d)]
    f2 = f[max_index:len(f)]
    return d1, f1, d2, f2


def seg_fitting(fmin,fmax,d,f):
    i = np.abs(f - fmin).argmin()   # 返回与某值相差最小值的索引
    j = np.abs(f - fmax).argmin()
    return d[i:j], f[i:j]


def fitting1(x, y):
    param_bounds = ([0, 0], [uplim0, uplim1])  # 指定参数的范围
    popt, pcov = curve_fit(func1, x, y, bounds=param_bounds)
    yvals = func1(x, popt[0], popt[1])
    # text_save("x1.txt",x)
    # text_save("y1.txt",yvals)
    r2 = round(np.corrcoef(y, yvals)[0, 1] ** 2, 4)
    # print(r2)
    lp1.append(popt[0])
    l01.append(popt[1])
    # print(popt)
    # print(r2)


def fitting2(x, y):
    param_bounds = ([0, 0, 0], [uplim0, uplim1, uplim2])  # 指定参数的范围
    popt, pcov = curve_fit(func2, x, y, bounds=param_bounds)
    yvals = func2(x, popt[0], popt[1], popt[2])
    # text_save("x2.txt", x)
    # text_save("y2.txt", yvals)
    r2 = round(np.corrcoef(y, yvals)[0, 1] ** 2, 4)
    # print(r2)
    lp2.append(popt[0])
    l02.append(popt[1])
    k0.append(popt[2])
    # print(popt)
    # print(r2)


def para2excel():
    df1 = pd.DataFrame(lp1, columns=['Lp_model1'])
    df2 = pd.DataFrame(l01, columns=['L0_model1'])
    df3 = pd.DataFrame(lp2, columns=['Lp_model2'])
    df4 = pd.DataFrame(l02, columns=['L0_model2'])
    df5 = pd.DataFrame(k0, columns=['k0_model2'])
    writer = pd.ExcelWriter("fitting parameters.xlsx")
    df1.to_excel(writer, 'Lp_model1')
    df2.to_excel(writer, 'L0_model1')
    df3.to_excel(writer, 'Lp_model2')
    df4.to_excel(writer, 'L0_model2')
    df5.to_excel(writer, 'k0_model2')
    writer.save()


def task_fit(txtname, force):
    mag_d, d = get_data(txtname)                # 读取数据
    magd_new, d_new = filter_nan(mag_d, d)      # 滤去nan
    magd_new = magd_new.astype(np.float64)      # 因为nan的存在，读入数据为str类型，转换成浮点数
    d_new = d_new.astype(np.float64)
    f = magd2f(magd_new, force)                 # 将磁铁距离换算成力
    d_stretch, f_stretch, d_reflex, f_reflex = segment(d_new, f)    # 拆分拉伸力与回复力
    # text_save("d1.txt",d_stretch)
    # text_save("f1.txt",f_stretch)
    # text_save("d2.txt",d_reflex)
    # text_save("f2.txt",f_reflex)
    d_model1, f_model1 = seg_fitting(0, 10, d_stretch, f_stretch)   # 拟合模型1
    d_model2, f_model2 = seg_fitting(3, 40, d_stretch, f_stretch)  # 拟合模型2
    fitting1(d_model1, f_model1)
    fitting2(f_model2, d_model2)


def mean_of_next(x):        # 求相邻两数的平均值
    y = []
    for i in range(0, len(x)-1):
        y.append((x[i]+x[i+1])/2)
    return np.array(y)


def func_guass(x, a, mu, sigma):
    return a*(np.exp(-(x-mu)**2/(2*sigma**2)))


def filter_para(x, a, b, upl):
    mu0 = np.mean(x)
    sigma0 = np.std(x)
    # 去除偏离大于b*sigma和过小的数据
    de = []
    for i in range(0, len(x)):
        if np.abs(mu0 - x[i]) > a * sigma0:  # 去除偏离过大的数据
            de.append(i)
        if x[i] < b:  # 去除过小数据
            de.append(i)
        if abs(x[i] - upl) < 3:  # 去除到达拟合上限的数据
            de.append(i)
    return de


def guass_fitting(x, de, pngname):
    xn = np.delete(x, de, axis=0)
    # guass拟合
    num_bins = 20           # 直方图柱子数量
    n, bins, patches = plt.hist(xn, num_bins, density=1, alpha=0.75)        # 绘制直方图
    gy = np.array(n)
    gx = mean_of_next(bins)
    param_bounds = ([0, 0, 0], [10, 5000, 50000])  # 指定参数的范围
    popt, pcov = curve_fit(func_guass, gx, gy, bounds=param_bounds)
    yvals = func_guass(gx, popt[0], popt[1], popt[2])
    r2 = round(np.corrcoef(gy, yvals)[0, 1] ** 2, 4)
    # plt.grid(True)          # 生成网格
    gx_new = np.linspace(gx.min(), gx.max(), 300)       # 绘制平滑曲线
    y_smooth = make_interp_spline(gx, yvals)(gx_new)
    plt.plot(gx_new, y_smooth)
    plt.xlabel('Values')
    plt.ylabel('Probability')
    plt.savefig(pngname)
    plt.close()
    if r2 < 0.6 or popt[1] < 0.1:
        return np.mean(xn), np.std(xn)
    return popt[1], popt[2]


def task_guass():
    # 矩阵化
    para = [lp1, l01, lp2, l02, k0]
    upl = [uplim0, uplim1, uplim0, uplim1, uplim2]
    pngname = ['lp1.png', 'l01.png', 'lp2.png', 'l02.png', 'k0.png']
    # 去除异常数据
    de = filter_para(para[0], 3, 0, upl[0])
    for i in range(1, len(para)):
        dei = filter_para(para[i], 3, 0, upl[i])
        de = list(set(de).union(set(dei)))          # 取并集
    # 高斯拟合
    mu = []
    sigma = []
    for i in range(0, len(para)):
        mui, sigmai = guass_fitting(para[i], de, pngname[i])
        mu.append(mui)
        sigma.append(sigmai)
    text_save("mu.txt", mu)
    text_save("sigma.txt", sigma)


def main():
    path = "file"
    count = 0
    for file in os.listdir(path):   # 计算总数
        count = count + 1
    for i in range(1, count+1):
        txtname = path + "\\" + str(i) + ".txt"
        force = np.array(pd.read_table("force.txt"))
        # print("\n" + "NO." + str(i))
        task_fit(txtname, force[i - 1])
    para2excel()                    # 将拟合参数写入Excel
    task_guass()
    print("计算完毕！")


main()

