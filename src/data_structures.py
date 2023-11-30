import torch
import torch.nn as nn
import torch.optim as optim


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# 表示客户的类，包含客户的客户编号、x坐标、y坐标、需求量、时间窗口开启时间、时间窗口关闭时间、服务所需时间。
class Customer:
    def __init__(self, id, x, y, demand, readyTime, dueTime, serviceTime):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.readyTime = readyTime
        self.dueTime = dueTime
        self.serviceTime = serviceTime


# 表示路径的类，包含路径经过的客户、载重量、距离、旅行时间、等待时间、延误时间。
class Route:
    def __init__(self, customers, distanceMatrix, timeMatrix):

        self.customers = customers
        self.load, self.distance, self.travelTime, self.waitTime, self.delayTime, self.delayTimeMatrix = self.__calculate(distanceMatrix, timeMatrix)

    # 计算距离、旅行时间、等待时间、延误时间
    def __calculate(self, distanceMatrix, timeMatrix):
        load = 0
        distance = 0
        travelTime = 0
        waitTime = 0
        delayTime = 0
        delayTimeMatrix = [0 for i in range(len(self.customers))]
        for i in range(len(self.customers) - 1):
            load += self.customers[i + 1].demand
            distance += distanceMatrix[self.customers[i].id][self.customers[i + 1].id]
            travelTime += timeMatrix[self.customers[i].id][self.customers[i + 1].id]
            if travelTime < self.customers[i + 1].readyTime:
                waitTime += self.customers[i + 1].readyTime - travelTime
                travelTime = self.customers[i + 1].readyTime
            elif travelTime > self.customers[i + 1].dueTime:
                delayTimeMatrix[i + 1] = travelTime - self.customers[i + 1].dueTime
                delayTime += travelTime - self.customers[i + 1].dueTime
            travelTime += self.customers[i + 1].serviceTime
        return load, distance, travelTime, waitTime, delayTime, delayTimeMatrix


# 表示解的类，包含解的路径、总距离、总旅行时间、总等待时间、总延误时间。
class Solution:
    def __init__(self, routes):
        self.routes = routes
        self.vehicleNumber = len(routes)
        self.distance, self.travelTime, self.waitTime, self.delayTime = self.__calculate()
        self.fitness = [self.vehicleNumber, self.distance]
        self.strength = 0
        self.r_fitness = 0
        self.f_fitness = 0

    # 计算总距离、总旅行时间、总等待时间、总延误时间
    def __calculate(self):
        distance = 0
        travelTime = 0
        waitTime = 0
        delayTime = 0
        for route in self.routes:
            distance += route.distance
            travelTime += route.travelTime
            waitTime += route.waitTime
            delayTime += route.delayTime
        return distance, travelTime, waitTime, delayTime
