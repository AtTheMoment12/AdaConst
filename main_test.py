import torch
import torchattacks
import torchattacks_new
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os
from tqdm import tqdm
import csv
import numpy as np
import timm
from pytorch_fid import fid_score
import pickle


def load_ground_truth(csv_filename):
    image_id_list = []
    label_list = []
    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_list.append(int(row['TrueLabel']) - 1)
    return image_id_list, label_list


def sensitivity(models, images, labels):
    adv_images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(labels.device)
    loss = nn.CrossEntropyLoss()
    adv_images.requires_grad = True
    outputs = models[0](adv_images)
    if len(models)>=2:
        for i in range(len(models)-1):
            outputs=outputs+models[i+1](adv_images)
    cost = loss(outputs, labels)
    grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
    s=torch.norm(grad, p=2, dim=(1, 2, 3))
    return s


def const(sens,epsilon,tao=0.99):
    upper_bound_value, _ = torch.kthvalue(sens, int(tao * len(sens)))
    sens_ = torch.clamp(sens, max=upper_bound_value)
    # 根据epsilon计算theta
    if epsilon <= 1:
        theta = 1
    elif 1 < epsilon <= 10:
        theta = 1 / epsilon
    else:
        theta = 0.1
    # 建立映射从[lb,up]到[theta,1]
    c = ((1 - theta) / (upper_bound_value - sens_.min())) * (upper_bound_value - sens_) + theta
    return c


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

# values are standard normalization for ImageNet images, from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([
    transforms.ToTensor(), ])


# fix the random seed of pytorch and make cudnn deterministic for reproducing the same results
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    # load image list
    image_id_list, label_list = load_ground_truth('./dataset/images.csv')
    total_number = len(image_id_list)
    # specify the device
    device = torch.device("cuda:0")

    ## target models
    model1 = models.inception_v3(pretrained=True, transform_input=False).eval()
    for param in model1.parameters():
        param.requires_grad = False
    model1.to(device)

    model2 = models.googlenet(pretrained=True, transform_input=False).eval()
    for param in model2.parameters():
        param.requires_grad = False
    model2.to(device)

    model3 = models.vgg16_bn(pretrained=True).eval()
    for param in model3.parameters():
        param.requires_grad = False
    model3.to(device)

    model4 = models.resnet152(pretrained=True).eval()
    for param in model4.parameters():
        param.requires_grad = False
    model4.to(device)

    model5 = models.mobilenet_v2(pretrained=True).eval()
    for param in model5.parameters():
        param.requires_grad = False
    model5.to(device)

    model6 = timm.create_model('vit_base_patch16_224',
                               checkpoint_path="C:/Users/10713/.cache/torch/hub/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth").eval()
    for param in model6.parameters():
        param.requires_grad = False
    model6.to(device)



    batch_size = 10
    num_batches = np.int(np.ceil(total_number / batch_size))

    # 第一阶段：计算约束强度
    '''sens=[]
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, total_number - k * batch_size)
        x_adv = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
        x_adv_224 = torch.zeros(batch_size_cur, 3, 224, 224).to(device)
        resize = transforms.Resize([224, 224])
        for i in range(batch_size_cur):
            x_adv[i] = trn(Image.open('./dataset/images' + '/' + image_id_list[k * batch_size + i] + '.png'))
            x_adv_224[i] = resize(x_adv[i])
        labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        s = sensitivity([model4], x_adv, labels)
        sens.append(s.tolist())
    sens=torch.tensor(sens)
    sens=sens.view(-1)
    with open("./sens.pkl", "wb") as f: 
        pickle.dump(sens, f) '''

    with open("./sens.pkl", "rb") as f:
       sens = pickle.load(f)


    '''pgd_data=[]
    for _, epsilon in enumerate([ 4,8,12,16,20, 24, 28,32]):
        # 第二阶段：对抗攻击
        output_path = './adv_images'
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        c=const(sens,epsilon)
        #atk = torchattacks_new.FGSM(model4, eps=epsilon / 255)
        atk = torchattacks_new.PGD(model4, eps=epsilon / 255, alpha=epsilon/255 / 10 * 2.5)

        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            # load a batch of input images with the size of batch_size*channel*height*width
            X_ori = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            for i in range(batch_size_cur):
                X_ori[i] = trn(Image.open(os.path.join('./dataset/images', image_id_list[k * batch_size + i]) + '.png'))
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
            c_k = c[k * batch_size:k * batch_size + batch_size_cur]

            X_adv = atk(X_ori, labels, c_k)

            # save the modified images
            for j in range(batch_size_cur):
                x_np = transforms.ToPILImage()(X_adv[j].detach().cpu())
                x_np.save(os.path.join(output_path, image_id_list[k * batch_size + j]) + '.png')

        # 第三阶段：计算黑盒攻击成功率ASR
        sr_1 = 0
        sr_2 = 0
        sr_3 = 0
        sr_4 = 0
        sr_5 = 0
        sr_6 = 0
        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            x_adv = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            x_adv_224 = torch.zeros(batch_size_cur, 3, 224, 224).to(device)
            resize = transforms.Resize([224, 224])
            for i in range(batch_size_cur):
                x_adv[i] = trn(Image.open(output_path + '/' + image_id_list[k * batch_size + i] + '.png'))
                x_adv_224[i] = resize(x_adv[i])
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

            pre_adv_1 = torch.argmax(model1((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_2 = torch.argmax(model2((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_3 = torch.argmax(model3(norm(x_adv)), dim=1)
            pre_adv_4 = torch.argmax(model4(norm(x_adv)), dim=1)
            pre_adv_5 = torch.argmax(model5(norm(x_adv)), dim=1)
            pre_adv_6 = torch.argmax(model6((x_adv_224 - 0.5) / 0.5), dim=1)

            sr_1 = sr_1 + sum((labels != pre_adv_1).float())
            sr_2 = sr_2 + sum((labels != pre_adv_2).float())
            sr_3 = sr_3 + sum((labels != pre_adv_3).float())
            sr_4 = sr_4 + sum((labels != pre_adv_4).float())
            sr_5 = sr_5 + sum((labels != pre_adv_5).float())
            sr_6 = sr_6 + sum((labels != pre_adv_6).float())

        print('attack_sucess_rate:', sr_1 / total_number, sr_2 / total_number, sr_3 / total_number,
              sr_4 / total_number, sr_5 / total_number, sr_6 / total_number,
              (sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number)

        ## 第四阶段：计算FID距离值
        fid_value = fid_score.calculate_fid_given_paths(['./dataset/images', output_path], 10, 'cuda:0', 2048,
                                                        num_workers=0)
        print('FID value:', fid_value)
        pgd_data.append((( (sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number).item(),fid_value.item()))
    print('pgd_data:',pgd_data)
    with open("./pgd_data.pkl", "wb") as f:
        pickle.dump(pgd_data, f)

    mi_data=[]
    for _, epsilon in enumerate([ 4,8,12,16,20, 24, 28,32]):
        # 第二阶段：对抗攻击
        output_path = './adv_images'
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        c=const(sens,epsilon)
        #atk = torchattacks_new.FGSM(model4, eps=epsilon / 255)
        atk = torchattacks_new.MIFGSM(model4, eps=epsilon / 255, alpha=epsilon/255 / 10 * 2.5)

        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            # load a batch of input images with the size of batch_size*channel*height*width
            X_ori = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            for i in range(batch_size_cur):
                X_ori[i] = trn(Image.open(os.path.join('./dataset/images', image_id_list[k * batch_size + i]) + '.png'))
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
            c_k = c[k * batch_size:k * batch_size + batch_size_cur]

            X_adv = atk(X_ori, labels, c_k)

            # save the modified images
            for j in range(batch_size_cur):
                x_np = transforms.ToPILImage()(X_adv[j].detach().cpu())
                x_np.save(os.path.join(output_path, image_id_list[k * batch_size + j]) + '.png')

        # 第三阶段：计算黑盒攻击成功率ASR
        sr_1 = 0
        sr_2 = 0
        sr_3 = 0
        sr_4 = 0
        sr_5 = 0
        sr_6 = 0
        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            x_adv = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            x_adv_224 = torch.zeros(batch_size_cur, 3, 224, 224).to(device)
            resize = transforms.Resize([224, 224])
            for i in range(batch_size_cur):
                x_adv[i] = trn(Image.open(output_path + '/' + image_id_list[k * batch_size + i] + '.png'))
                x_adv_224[i] = resize(x_adv[i])
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

            pre_adv_1 = torch.argmax(model1((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_2 = torch.argmax(model2((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_3 = torch.argmax(model3(norm(x_adv)), dim=1)
            pre_adv_4 = torch.argmax(model4(norm(x_adv)), dim=1)
            pre_adv_5 = torch.argmax(model5(norm(x_adv)), dim=1)
            pre_adv_6 = torch.argmax(model6((x_adv_224 - 0.5) / 0.5), dim=1)

            sr_1 = sr_1 + sum((labels != pre_adv_1).float())
            sr_2 = sr_2 + sum((labels != pre_adv_2).float())
            sr_3 = sr_3 + sum((labels != pre_adv_3).float())
            sr_4 = sr_4 + sum((labels != pre_adv_4).float())
            sr_5 = sr_5 + sum((labels != pre_adv_5).float())
            sr_6 = sr_6 + sum((labels != pre_adv_6).float())

        print('attack_sucess_rate:', sr_1 / total_number, sr_2 / total_number, sr_3 / total_number,
              sr_4 / total_number, sr_5 / total_number, sr_6 / total_number,
              (sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number)

        ## 第四阶段：计算FID距离值
        fid_value = fid_score.calculate_fid_given_paths(['./dataset/images', output_path], 10, 'cuda:0', 2048,
                                                        num_workers=0)
        print('FID value:', fid_value)
        mi_data.append((( (sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number).item(),fid_value.item()))
    print('mi_data:',mi_data)
    with open("./mi_data.pkl", "wb") as f:
        pickle.dump(mi_data, f)'''

    mil1mask_data = []
    for epsilon in [100000,300000,900000,1500000,2500000,3500000]:
        # 第二阶段：对抗攻击
        output_path = './adv_images'
        if os.path.exists(output_path) == False:
            os.makedirs(output_path)
        c = const(sens, epsilon/100000)
        # atk = torchattacks_new.FGSM(model4, eps=epsilon / 255)
        atk = torchattacks_new.MIFGSML1MASK(model4, eps=epsilon / 255, alpha=(epsilon / 255 /270000/50)**0.5, steps=10)

        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            # load a batch of input images with the size of batch_size*channel*height*width
            X_ori = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            for i in range(batch_size_cur):
                X_ori[i] = trn(Image.open(os.path.join('./dataset/images', image_id_list[k * batch_size + i]) + '.png'))
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
            c_k = c[k * batch_size:k * batch_size + batch_size_cur]

            X_adv = atk(X_ori, labels, c_k)

            # save the modified images
            for j in range(batch_size_cur):
                x_np = transforms.ToPILImage()(X_adv[j].detach().cpu())
                x_np.save(os.path.join(output_path, image_id_list[k * batch_size + j]) + '.png')

        # 第三阶段：计算黑盒攻击成功率ASR
        sr_1 = 0
        sr_2 = 0
        sr_3 = 0
        sr_4 = 0
        sr_5 = 0
        sr_6 = 0
        for k in tqdm(range(0, num_batches)):
            batch_size_cur = min(batch_size, total_number - k * batch_size)
            x_adv = torch.zeros(batch_size_cur, 3, 299, 299).to(device)
            x_adv_224 = torch.zeros(batch_size_cur, 3, 224, 224).to(device)
            resize = transforms.Resize([224, 224])
            for i in range(batch_size_cur):
                x_adv[i] = trn(Image.open(output_path + '/' + image_id_list[k * batch_size + i] + '.png'))
                x_adv_224[i] = resize(x_adv[i])
            labels = torch.tensor(label_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)

            pre_adv_1 = torch.argmax(model1((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_2 = torch.argmax(model2((x_adv - 0.5) / 0.5), dim=1)
            pre_adv_3 = torch.argmax(model3(norm(x_adv)), dim=1)
            pre_adv_4 = torch.argmax(model4(norm(x_adv)), dim=1)
            pre_adv_5 = torch.argmax(model5(norm(x_adv)), dim=1)
            pre_adv_6 = torch.argmax(model6((x_adv_224 - 0.5) / 0.5), dim=1)

            sr_1 = sr_1 + sum((labels != pre_adv_1).float())
            sr_2 = sr_2 + sum((labels != pre_adv_2).float())
            sr_3 = sr_3 + sum((labels != pre_adv_3).float())
            sr_4 = sr_4 + sum((labels != pre_adv_4).float())
            sr_5 = sr_5 + sum((labels != pre_adv_5).float())
            sr_6 = sr_6 + sum((labels != pre_adv_6).float())

        print('attack_sucess_rate:', sr_1 / total_number, sr_2 / total_number, sr_3 / total_number,
              sr_4 / total_number, sr_5 / total_number, sr_6 / total_number,
              (sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number)

        ## 第四阶段：计算FID距离值
        fid_value = fid_score.calculate_fid_given_paths(['./dataset/images', output_path], 10, 'cuda:0', 2048,
                                                        num_workers=0)
        print('FID value:', fid_value)
        mil1mask_data.append((((sr_1 + sr_2 + sr_3 + sr_5 + sr_6) / 5 / total_number).item(), fid_value.item()))

    print('mil1mask_data:', mil1mask_data)
    with open("./mil1mask_data.pkl", "wb") as f:
        pickle.dump(mil1mask_data, f)