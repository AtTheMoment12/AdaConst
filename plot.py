###4.步长系数影响图
import matplotlib.pyplot as plt

from plotAPGD import x_adv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#步长研究横轴alpha=1-10；y1表示asr;y2表示FID
x=[1,1.5,2,2.5,3]
#y1=[13.68,16.98,17.44,17,16.66]
#y2=[5.76,11.14,11.64,11.21,10.78]

y1=[13.78,17.7,17.4,17.1,17.31]
y2=[6.05,12.05,12.38,12.37,12.49]

fig, ax1 = plt.subplots()

ax1.plot(x, y1, "ob-",  label="ASR")
ax1.set_xlabel("步长系数")
ax1.set_ylabel("ASR/%")

ax2 = ax1.twinx()
ax2.plot(x, y2, "*r-", label="FID")
ax2.set_ylabel("FID")

fig.legend(loc="upper right", bbox_to_anchor=(1,0.2), bbox_transform=ax1.transAxes)
ax1.grid(b=True, which='both', axis='both',)
plt.show()



###FID_ASR关系图

import matplotlib.pyplot as plt
import numpy as np

# L1
points_l1_ = [(26.41,24.64),(45.26,35.76),(59.2,42.62),(73.31,49.7),(100.24,61.82),(125.22,70.36)]
points_l1 = [(38.85,26.98),(53.98,34.68),(65.36,40.98),(75.28,46.42),(93.19,55.64),(135,72.56)]
# 绘制折线图
plt.plot([p[0] for p in points_l1_], [p[1] for p in points_l1_], 'r-', label='L1')
# 绘制五角星标记
for point in points_l1_:
    plt.scatter(*point, marker='*', color='r')

# L1MASK
points_l1mask_ = [(33.04,28.8),(67.22,47.96),(89.81,58.86),(119.44,70.18),(143.28,76.5)]
points_l1mask = [(32.19,27.24),(61.72,40.78),(81.72,51.64),(109.03,64.16),(132.2,72.14)]
# 绘制折线图
plt.plot([p[0] for p in points_l1mask_], [p[1] for p in points_l1mask_], 'c-', label='L1mask')
# 绘制五角星标记
for point in points_l1mask_:
    plt.scatter(*point, marker='h', color='c')

#PMR
points_pmr_ = [(50.97,39.2),(80.47,53.66),(103.33,63.96)]
# 绘制折线图
plt.plot([p[0] for p in points_pmr_], [p[1] for p in points_pmr_], 'c-', label='pmr')
# 绘制五角星标记
for point in points_pmr_:
    plt.scatter(*point, marker='o', color='g')


# L2
points_l2_ = [(33.06,28.18),(65.12,45.6),(91.02,57.76),(113.41,66.42),(134.65,72.5)]
points_l2 = [(37.11,25.92),(65.99,40.16),(89.78,52.18),(112.47,62.44),(131.1,69.44)]
# 绘制折线图
plt.plot([p[0] for p in points_l2], [p[1] for p in points_l2], 'g-', label='L2')
# 绘制圆点标记
for point in points_l2:
    plt.scatter(*point, marker='o', color='g')

# Linf
points_linf_ = [(25.74,23.16),(57.18,40.26),(82.27,52.04),(103.79,60.78),(122.1,67.34),(139.12,72.22)]
points_linf = [(30.78,23.2),(59.07,36.36),(79.83,46.24),(95.99,54.12),(111.26,59.44),(135.59,69.32)]
# 绘制折线图
plt.plot([p[0] for p in points_linf], [p[1] for p in points_linf], 'b-', label='Linf')
# 绘制方块标记
for point in points_linf:
    plt.scatter(*point, marker='s', color='b')

# 添加图例
plt.legend()
plt.xlabel('FID')
plt.ylabel('attack_sucess_rate')
# 显示图形
plt.show()


## fgsm--fgsm_acaconst对比图

import matplotlib.pyplot as plt

#fgsm = [(0.2210,23.31),(0.3380,46.9316),(0.4298,64.2446),(0.5020,79.6108),(0.5650,93.1703),(0.6184,106.3759),(0.6592,118.48)]
fgsm = [(0.3380,46.9316),(0.4298,64.2446),(0.5020,79.6108),(0.5650,93.1703),(0.6184,106.3759)]
plt.plot([p[0] for p in fgsm], [p[1] for p in fgsm], 'b-', label='FGSM', marker='*')

#fgsm_a = [(0.1762,15.57),(0.2668,33.36),(0.3484,48.28),(0.4194,61.09),(0.4816,72.71),(0.5332,83.42),(0.5806,94.18)]
fgsm_a = [(0.3484,48.28),(0.4194,61.09),(0.4816,72.71),(0.5332,83.42),(0.5806,94.18)]
plt.plot([p[0] for p in fgsm_a], [p[1] for p in fgsm_a], 'r-', label='FGSM_adaconst', marker='s')
# 添加图例
plt.legend()
plt.xlabel('attack_sucess_rate')
plt.ylabel('FID')
# 显示图形
plt.show()

## MI--MI_acaconst对比图

import matplotlib.pyplot as plt
mila=[(0.1834000200033188, 13.326042827498952), (0.2600000202655792, 27.687536492248), (0.4174000024795532, 54.1609617272544), (0.5278000235557556, 74.35918100860624), (0.639400064945221, 100.47527440691647), (0.7156000733375549, 121.75508348365423)]
#mi=[(0.2316,25.74),(0.4026,57.18),(0.5204,82.27),(0.6078,103.79),(0.6734,122.1)]
plt.plot([p[0] for p in mila], [p[1] for p in mila], 'b-', label='MILA', marker='*')
DeCoWA=[(0.3150,38.7113),(0.5658,87.4220), (0.7028,118.3632),(0.7754,141.2777),(0.8214,156.9629), (0.8554,172.9909)]

DeCoWA=[(0.3150,38.7113),(0.5658,87.4220), (0.7028,119.3632),(0.7754,141.2777),(0.8214,156.9629), (0.8554,172.9909)]
#mi_a = [ (0.3118000328540802, 39.670027015152755), (0.4248000383377075, 60.59906646953738), (0.516200065612793, 78.73300092360438), (0.592400074005127, 95.13199711416019), (0.6540000534057617, 113.1016025575646)]
plt.plot([p[0] for p in DeCoWA], [p[1] for p in DeCoWA], 'r-', label='DeCoWA', marker='s')
# 添加图例
plt.legend()
plt.xlabel('attack_sucess_rate')
plt.ylabel('FID')
# 显示图形
plt.show()

## pgd--pgd_acaconst对比图

import matplotlib.pyplot as plt

pgd=[(0.2872,38.84),(0.3736,57.27),(0.4562,75.93),(0.5174,91.42),(0.5718,106.55)]
plt.plot([p[0] for p in pgd], [p[1] for p in pgd], 'b-', label='PGD', marker='*')

pgd_a = [(0.2948000431060791, 40.270286854904555), (0.3680000305175781, 53.72142582004011), (0.4254000186920166, 66.153112563592), (0.4864000082015991, 81.66758233968255), (0.542400062084198, 93.45258847318036), (0.5800000429153442, 103.63552408732659)]
plt.plot([p[0] for p in pgd_a], [p[1] for p in pgd_a], 'r-', label='PGD_adaconst', marker='s')
# 添加图例
plt.legend()
plt.xlabel('attack_sucess_rate')
plt.ylabel('FID')
# 显示图形
plt.show()

##线性插值

import numpy as np
fgsm = [(0.2210,23.31),(0.3380,46.9316),(0.4298,64.2446),(0.5020,79.6108),(0.5650,93.1703),(0.6184,106.3759),(0.6592,118.48)]
fgsm_a = [(0.1762,15.57),(0.2668,33.36),(0.3484,48.28),(0.4194,61.09),(0.4816,72.71),(0.5332,83.42),(0.5806,94.18)]

x=[]
y=[]
for _,p in enumerate(fgsm):
    x.append(p[0])
    y.append(p[1])
y_new = np.interp([0.40,0.45,0.50,0.55,0.60] , x, y)
y_new

x_ = []
y_ = []
for _, p in enumerate(fgsm_a):
    x_.append(p[0])
    y_.append(p[1])
y_new_ = np.interp([0.35,0.40,0.45,0.50,0.55,0.60], x_, y_)




import matplotlib.pyplot as plt

'''theta_1=[(0.1674,14.03),(0.264,32.75),(0.3484,48.28),(0.4194,61.09),(0.4816,72.71),(0.5332,83.42),(0.5806,94.18)]
theta_3=[(0.1806,16.29),(0.2844,36.37),(0.3666,52.52),(0.445,65.88),(0.5048,77.99),(0.5598,89.53),(0.604,100.92)]
theta_5=[(0.1914,18.28),(0.2992,39.58),(0.386,56.14),(0.4654,70.2),(0.5228,82.76),(0.5772,94.66),(0.6246,106.42)]
theta_7=[(0.1992,19.51),(0.313,42.51),(0.4054,59.42),(0.4786,73.83),(0.5384,86.89),(0.597,99.21),(0.6392,111.5)]
theta_9=[(0.2004,19.79),(0.3222,44.17),(0.4192,62.04),(0.4928,77.21),(0.5552,90.71),(0.6096,103.5),(0.6514,115.9)]
theta_10=[(0.2210,23.31),(0.3380,46.93),(0.4298,64.24),(0.5020,79.61) ,(0.5650,93.17), (0.6184,106.38 ),(0.6592,118.48)]
theta_01=[(0.165,13.34),(0.2536,31.28),(0.3386,46.34),(0.4068,58.85),(0.4702,69.99),(0.52,80.59),(0.5666,90.77)]'''
theta_01=[(0.3386,46.34),(0.4068,58.85),(0.4702,69.99),(0.52,80.59),(0.5666,90.77)]
theta_1=[(0.3484,48.28),(0.4194,61.09),(0.4816,72.71),(0.5332,83.42),(0.5806,94.18)]
theta_3=[(0.3666,52.52),(0.445,65.88),(0.5048,77.99),(0.5598,89.53),(0.604,100.92)]
theta_7=[(0.313,42.51),(0.4054,59.42),(0.4786,73.83),(0.5384,86.89),(0.597,99.21)]
theta_10=[(0.3380,46.93),(0.4298,64.24),(0.5020,79.61) ,(0.5650,93.17), (0.6184,106.38 )]

plt.plot([p[0] for p in theta_01], [p[1] for p in theta_01], 'b-', label=r'$\theta$ =0.01', marker='D', markersize=4)
plt.plot([p[0] for p in theta_1], [p[1] for p in theta_1], 'r-', label=r'$\theta$ =0.1', marker='o', markersize=4)
plt.plot([p[0] for p in theta_3], [p[1] for p in theta_3], 'g-', label=r'$\theta$ =0.3', marker='s', markersize=4)
#plt.plot([p[0] for p in theta_5], [p[1] for p in theta_5], 'b-', label=r'$\theta$ =0.5', marker='D')
plt.plot([p[0] for p in theta_7], [p[1] for p in theta_7], 'c-', label=r'$\theta$ =0.7', marker='*', markersize=4)
#plt.plot([p[0] for p in theta_9], [p[1] for p in theta_9], 'm-', label=r'$\theta$ =0.9', marker='+')
plt.plot([p[0] for p in theta_10], [p[1] for p in theta_10], 'k-', label=r'$\theta$ =1', marker='x', markersize=4)
# 添加图例
plt.legend()
plt.xlabel('attack_sucess_rate')
plt.ylabel('FID')
# 显示图形
plt.show()


import matplotlib.pyplot as plt
MI_LA=[(0.1834000200033188, 13.326042827498952), (0.2600000202655792, 27.687536492248), (0.4174000024795532, 54.1609617272544), (0.5278000235557556, 74.35918100860624), (0.639400064945221, 100.47527440691647), (0.7156000733375549, 121.75508348365423)]
Admix=[(0.2294,22.99) ,(0.3880,54.17) ,(0.5140,80.40),(0.6104,102.01),(0.6726,122.46) ,(0.7170,137.32) ]
FIA=[(0.2208,23.56),(0.3892,55.34), (0.5026,82.22), (0.6028,101.78), (0.6620,120.04), (0.7170,134.79) ]
ILPD=[(0.2768,29.82),(0.5224,73.91), (0.6670,73.91),(0.7436,128.05) ,(0.8002,145.15) ,(0.8362,160.15)]
APGD=[(0.1450,12.90), (0.2234,29.08), (0.2988,43.80), (0.3524,53.57), (0.4020,65.36), (0.4360,73.90),(0.4806,88.55)]
Autoattack=[(0.2252,26.98),(0.2972,40.48),(0.4220,66.88),(0.5346,105.02)]
#DeCoWA=[(0.3150,38.7113),(0.5658,87.4220), (0.7028,118.3632),(0.7754,141.2777),(0.8214,156.9629), (0.8554,172.9909)]
DeCoWA=[(0.3150,38.7113),(0.5658,87.4220), (0.7028,118.3632),(0.7754,141.2777),(0.8214,156.9629), (0.8554,172.9909)]
plt.plot([p[0] for p in MI_LA], [p[1] for p in MI_LA], 'r-', label=r'MI-LA', marker='D', markersize=4)
plt.plot([p[0] for p in Admix], [p[1] for p in Admix], 'b-', label=r'Admix', marker='o', markersize=4)
plt.plot([p[0] for p in FIA], [p[1] for p in FIA], 'g-', label=r'FIA', marker='s', markersize=4)
plt.plot([p[0] for p in APGD], [p[1] for p in APGD], 'c-', label=r'APGD', marker='*', markersize=4)
plt.plot([p[0] for p in Autoattack], [p[1] for p in Autoattack], 'k-', label=r'Autoattack', marker='x', markersize=4)
plt.plot([p[0] for p in DeCoWA], [p[1] for p in DeCoWA],'m-', label=r'DeCoWA', marker='v', markersize=4)
# 添加图例
plt.legend()
plt.xlabel('attack_sucess_rate')
plt.ylabel('FID')
# 显示图形
plt.show()

#绘制3种范数归一化的范数对比
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

plt.plot(l1.cpu().numpy(), 'g-',marker='o',label='L1')
plt.plot(l2.cpu().numpy(), 'r-',marker='s',label='L2')
plt.plot(linf.cpu().numpy(), 'b-',marker='*',label='Linf')
plt.xticks([0, 1, 2, 3, 4, 5,6,7,8,9], ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.xlabel("样本")
plt.ylabel("归一化后的范数")
plt.legend()
plt.show()

## 6个model显著性对比图

plt.plot(s1.cpu().numpy(), 'r-')
plt.plot(s2.cpu().numpy(), 'o-')
plt.plot(s3.cpu().numpy(), 'y-')
plt.plot(s4.cpu().numpy(), 'g-')
plt.plot(s5.cpu().numpy(), 'b-')
plt.plot(s6.cpu().numpy(), 'k-')


###图2 无穷范数（上）和1范数（下）投影对抗噪声分析   图4 基于L1-mask约束的对抗噪声分析

import torch
from torchvision import models, transforms
from PIL import Image

trn = transforms.Compose([
        transforms.ToTensor(), ])

x=Image.open('./image_compair/0c7ac4a8c9dfa802_.png')
x_01=trn(x)
x=Image.open('./image_compair/01dd15caa1b2c7b4_.png')
x_02=trn(x)
x=Image.open('./image_compair/01244097ca8ffdfa_.png')
x_03=trn(x)

x=Image.open('./image_compair/0c7ac4a8c9dfa802_1.png')
x_11=trn(x)
x=Image.open('./image_compair/01dd15caa1b2c7b4_1.png')
x_12=trn(x)
x=Image.open('./image_compair/01244097ca8ffdfa_1.png')
x_13=trn(x)

x=Image.open('./image_compair/0c7ac4a8c9dfa802_1m.png')
x_1mask1=trn(x)
x=Image.open('./image_compair/01dd15caa1b2c7b4_1m.png')
x_1mask2=trn(x)
x=Image.open('./image_compair/01244097ca8ffdfa_1m.png')
x_1mask3=trn(x)

x=Image.open('./image_compair/0c7ac4a8c9dfa802_2.png')
x_21=trn(x)
x=Image.open('./image_compair/01dd15caa1b2c7b4_2.png')
x_22=trn(x)
x=Image.open('./image_compair/01244097ca8ffdfa_2.png')
x_23=trn(x)

x=Image.open('./image_compair/0c7ac4a8c9dfa802_inf.png')
x_inf1=trn(x)
x=Image.open('./image_compair/01dd15caa1b2c7b4_inf.png')
x_inf2=trn(x)
x=Image.open('./image_compair/01244097ca8ffdfa_inf.png')
x_inf3=trn(x)

noise11=x_11-x_01
noise12=x_12-x_02
noise13=x_13-x_03
noise1=torch.cat((noise11, noise12, noise13), dim=0)
noise1mask1=x_1mask1-x_01
noise1mask2=x_1mask2-x_02
noise1mask3=x_1mask3-x_03
noise1mask=torch.cat((noise1mask1, noise1mask2, noise1mask3), dim=0)
noise21=x_21-x_01
noise22=x_22-x_02
noise23=x_23-x_03
noise2=torch.cat((noise21, noise22, noise23), dim=0)
noiseinf1=x_inf1-x_01
noiseinf2=x_inf2-x_02
noiseinf3=x_inf3-x_03
noiseinf=torch.cat((noiseinf1, noiseinf2, noiseinf3), dim=0)

#绘制直方图
# 计算tensor的绝对值大小
abs_tensor = torch.abs(noise1mask1.view(3*299*299))
# 将tensor转换为numpy数组
abs_numpy = abs_tensor.cpu().numpy()
plt.figure(figsize=(5,4))
plt.hist(abs_numpy, bins=10, rwidth=0.8, align='left')
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='both')
plt.xlabel('参数绝对值',fontsize=9)
plt.ylabel('频数',fontsize=9)
plt.ylim((0, 300000))
plt.xlim((0,6))

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 使用matplotlib绘制直方图
plt.hist(sensitivity.numpy(), bins=30, alpha=0.5, color='blue', edgecolor='black')

# 设置坐标轴标签
plt.xlabel('敏感度值')
plt.ylabel('频数')

# 显示图表
plt.show()


#绘制热力图
import torch
import matplotlib.pyplot as plt
# 将tensor转换为numpy数组
array = noise11.abs().mean(dim=0).cpu().numpy()
# 绘制热力图
plt.imshow(array, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

#绘制图片
from PIL import Image
import matplotlib.pyplot as plt

# 将Tensor的形状从 (C, H, W) 转换为 (H, W, C)
tensor = x_adv[0].permute(1, 2, 0)
# 假设 tensor 的值在 [0, 1] 范围内，将值缩放到 [0, 255]
tensor = tensor * 255
# 将Tensor的数据类型转换为uint8
tensor = tensor.to(torch.uint8)
# 将Tensor转换为PIL图像
image = Image.fromarray(tensor.cpu().numpy())

# 显示图片
plt.imshow(image)
plt.axis('off')  # 不显示坐标轴
plt.show()




import matplotlib.pyplot as plt
import numpy as np
# 设置图像宽度、高度和分辨率
width = 7.5 / 2.54 /2 # 厘米转换为英寸
height = width*.618   # 黄金分割比例
dpi = 400
fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

# 数据
data = [3.5, 2.8, 2.9, 4.3, 2.7, 3.5, 3.7, 4.1, 4.2, 4.3, 64]
labels = np.linspace(0, 0.06, 11)
x = np.arange(len(labels))
# 绘制柱状图
ax.bar(x, data)

# 设置横坐标标签和纵坐标标签
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('参数绝对值', fontsize=9, fontname='SimHei')
ax.set_ylabel('频率/%', fontsize=9, fontname='SimHei')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 显示图像
plt.show()


noise1np = transforms.ToPILImage()(noise1.detach().cpu())
noise1np.save('./demo/noisel1.png')

noiseinfnp = transforms.ToPILImage()(noiseinf.detach().cpu())
noiseinfnp.save('./demo/noiseinf.png')




import torch
import matplotlib.pyplot as plt
# 计算tensor的绝对值大小
abs_tensor = torch.abs(noise1.view(noise1.numel()))
# 将tensor转换为numpy数组
abs_numpy = abs_tensor.cpu().numpy()
# 使用matplotlib绘制分布图
plt.hist(abs_numpy, bins=20)
plt.xlabel('绝对值大小')
plt.ylabel('频数')
plt.title('绝对值大小分布图')
plt.show()

###检测图片置信度
X_ori = torch.zeros(1, 3, 299, 299).to(device)
X_ori[0] = trn(Image.open(os.path.join('./dataset/images', image_id_list[707]) + '.png'))
labels = torch.tensor(label_list[707:708]).to(device)
atk = torchattacks_new.MIFGSM(model4, eps=4 / 255, alpha=1/255)
X_adv = atk(X_ori, labels, torch.tensor(1.0))
out = model4(X_adv)
probs = torch.nn.functional.softmax(out, dim=1)
confidence, class_index = torch.max(probs, dim=1)

out = model4(X_ori)
probs = torch.nn.functional.softmax(out, dim=1)
confidence, class_index = torch.max(probs, dim=1)
confidence



X_ori2 = torch.zeros(1, 3, 299, 299).to(device)
X_ori2[0] = trn(Image.open(os.path.join('./dataset/images', image_id_list[448]) + '.png'))
labels2 = torch.tensor(label_list[448:449]).to(device)
atk = torchattacks_new.MIFGSM(model4, eps=1 / 255, alpha=0.25/255)
X_adv2 = atk(X_ori2, labels2, torch.tensor(1.0))
out2 = model4(X_adv2)
probs2 = torch.nn.functional.softmax(out2, dim=1)
confidence2, class_index2 = torch.max(probs2, dim=1)

out = model4(X_ori2)
probs = torch.nn.functional.softmax(out, dim=1)
confidence, class_index = torch.max(probs, dim=1)
confidence

#绘制图片
from PIL import Image
import matplotlib.pyplot as plt

# 将Tensor的形状从 (C, H, W) 转换为 (H, W, C)
tensor = X_adv2.permute(1, 2, 0)
# 假设 tensor 的值在 [0, 1] 范围内，将值缩放到 [0, 255]
tensor = tensor * 255
# 将Tensor的数据类型转换为uint8
tensor = tensor.to(torch.uint8)
# 将Tensor转换为PIL图像
image = Image.fromarray(tensor.cpu().numpy())

# 显示图片
plt.imshow(image)
plt.axis('off')  # 不显示坐标轴
plt.show()