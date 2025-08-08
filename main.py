import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time 
import tqdm
import numpy as np
from ssndataSet import MyDateSet
from ssnModel import Multi_Branch
from metrics import get_regression_metrics
from myTest import plotsCatter, plotBar


start_time = time.time()

num_epochs = 300
trainModel = True
modelname = 'wyzmodel.pth'
root_dir = r'E:\SSN\medician_content\test\Output.xlsx'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = MyDateSet(root_dir, model="Train", transform=transforms.Compose([transforms.ToTensor()])) 
val_dataset = MyDateSet(root_dir, model="Val", transform=transforms.ToTensor())
test_dataset = MyDateSet(root_dir, model="Test", transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False) 
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = Multi_Branch(102, 102, 102, 102, 3)

criterion = nn.MSELoss()

optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1) 

if trainModel:
    lossMin = 2
    for epoch in range(num_epochs):
        model.train() 
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]')             
        for step, (img, img1, img2, img3, label) in t:
            output = model(img.to(device), img1.to(device), img2.to(device), img3.to(device))
            loss = criterion(output, label.to(device))
            lossTatol += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
        lossAverage = lossTatol/(step+1)
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lr: {}'.format(epoch+1, num_epochs, lossAverage, loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))

        model.eval()

        lossTatol = 0
        t = tqdm.tqdm(enumerate(val_loader),desc = f'[Test]') 
        for step, (img, img1, img2, img3, label) in t:
            output = model(img.to(device), img1.to(device), img2.to(device), img3.to(device))
            loss = criterion(output, label.to(device))
            lossTatol += loss.item()

        lossAverage = lossTatol/(step+1)

        if lossMin > lossAverage:
            lossMin = lossAverage
            torch.save(model.state_dict(), modelname)
            print('Model Saved!! lossMin: {:.4f}'.format(lossMin))
        else:
            print('lossMin: {:.4f}, lossNow: {:.4f}'.format(lossMin, lossAverage))
        print(' ')
        
model.load_state_dict(torch.load(modelname))

model.eval()

true_list1 = []
output_list1 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(train_loader),desc = f'[Train]') 
for step, (img, img1, img2, img3, label) in t:
    output = model(img.to(device), img1.to(device), img2.to(device), img3.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list1.append(label.cpu().detach().numpy()[0])
    output_list1.append(output.cpu().detach().numpy()[0])
lossAverage = lossTatol/(step+1)

true_list1 = np.array(true_list1)
output_list1 = np.array(output_list1)
print(get_regression_metrics(true_list1, output_list1))
print(get_regression_metrics(true_list1[:, 0]*12, output_list1[:, 0]*12))
print(get_regression_metrics(true_list1[:, 1]*4.1, output_list1[:, 1]*4.1))
print(get_regression_metrics(true_list1[:, 2]*60, output_list1[:, 2]*60))
print('lossNow: {:.10f}'.format(lossAverage))

true_list2 = []
output_list2 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') 
for step, (img, img1, img2, img3, label) in t:
    output = model(img.to(device), img1.to(device), img2.to(device), img3.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list2.append(label.cpu().detach().numpy()[0])
    output_list2.append(output.cpu().detach().numpy()[0])
lossAverage = lossTatol/(step+1)

true_list2 = np.array(true_list2)
output_list2 = np.array(output_list2)
print(get_regression_metrics(true_list2, output_list2))
print(get_regression_metrics(true_list2[:, 0]*12, output_list2[:, 0]*12))
print(get_regression_metrics(true_list2[:, 1]*4.1, output_list2[:, 1]*4.1))
print(get_regression_metrics(true_list2[:, 2]*60, output_list2[:, 2]*60))
print('lossNow: {:.10f}'.format(lossAverage))

true_list3 = []
output_list3 = []

lossTatol = 0
t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') 
for step, (img, img1, img2, img3, label) in t:
    output = model(img.to(device), img1.to(device), img2.to(device), img3.to(device))
    loss = criterion(output, label.to(device))
    lossTatol += loss.item()
    true_list3.append(label.cpu().detach().numpy()[0])
    output_list3.append(output.cpu().detach().numpy()[0])
lossAverage = lossTatol/(step+1)

true_list3 = np.array(true_list3)
output_list3 = np.array(output_list3)
result_test = get_regression_metrics(true_list3, output_list3)
print(result_test)
result_test1 = get_regression_metrics(true_list3[:, 0]*12, output_list3[:, 0]*12)
print(result_test1)
result_test2 = get_regression_metrics(true_list3[:, 1]*4.1, output_list3[:, 1]*4.1)
print(result_test2)
result_test3 = get_regression_metrics(true_list3[:, 2]*60, output_list3[:, 2]*60)
print(result_test3)
print('lossNow: {:.10f}'.format(lossAverage))

final_time = time.time() - start_time
print(f'Final Time: {final_time:.2f} seconds')

# plotsCatter(true_list3[:, 0]*12, output_list3[:, 0]*12, 'Flavonoids(mg/g)', result_test1[0], result_test1[1], result_test1[2], result_test1[3], [0,4,8,12,16], [0,16], 9.5, 4.7, 1.5, 1.5, 1.5)
# plotsCatter(true_list3[:, 1]*4.1, output_list3[:, 1]*4.1, 'Saponins(mg/g)', result_test2[0], result_test2[1], result_test2[2], result_test2[3], [1,2,3,4,5], [0,5], 2.95, 1.4, 0.45, 0.45, 0.45)
# plotsCatter(true_list3[:, 2]*60, output_list3[:, 2]*60, 'Polysaccharides(mg/g)', result_test3[0], result_test3[1], result_test3[2], result_test3[3], [20,30,40,50,60,70], [0,70], 41, 20.1, 6.5, 6.5, 6.5)

# plotBar(true_list3[:, 0]*12, output_list3[:, 0]*12, [5, 10], '(a)Flavonoids')
# plotBar(true_list3[:, 1]*4.1, output_list3[:, 1]*4.1, [2, 4], '(b)Saponins')
# plotBar(true_list3[:, 2]*60, output_list3[:, 2]*60, [20, 40], '(c)Polysaccharides')
