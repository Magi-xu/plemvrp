import os
import pickle
import time

from solver import *
from utils import *

benchmarks = ["c101"]
for data in benchmarks:
    # 读取数据
    dataset = "../data/solomon_100/" + data + ".txt"
    name, vehicleNumber, vehicleCapacity, customers = read_vrptw_dataset(dataset)

    # 车辆速度
    VEHICLE_SPEED = 1
    # 车辆容量
    VEHICLE_CAPACITY = vehicleCapacity
    # 最大允许的延误时间
    MAX_DELAY_TIME = 0
    # 最大允许的返程时间
    MAX_RETURN_TIME = customers[0].dueTime
    # 种群大小
    POPULATION_SIZE = 200
    # 迭代次数
    ITERATIONS = 400
    # k
    k = math.sqrt(3 * POPULATION_SIZE)
    # mlp超参数
    input_size = len(customers) - 1
    hidden_size = 128
    output_size = 2
    learning_rate = 0.001
    epochs = 100
    # 初始化模型、损失函数和优化器
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 获取表示客户之间距离的矩阵
    distanceMatrix = getDistanceMatrix(customers)
    # 获取表示客户之间时间的矩阵
    timeMatrix = getTimeMatrix(customers, distanceMatrix, VEHICLE_SPEED)

    # 种群初始化
    print("种群初始化...", end=" ")
    population = random_pfihs_init(POPULATION_SIZE / 2, customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix)
    population.extend(init_population(POPULATION_SIZE / 2, customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix))
    print("\t完成")

    population_code = []

    best_solution = None

    for n_iter in range(ITERATIONS):
        # 计算种群中每个个体的强度值
        population = calculate_strength(population)
        # 计算原始适应度
        population = calculate_r_fitness(population)
        # 获取拥挤度矩阵
        fitness_distanceMatrix = calculate_distance_matrix(population)
        # 计算拥挤度适应度
        population = calculate_f_fitness(population, fitness_distanceMatrix, k)
        # 根据拥挤度适应度对种群进行排序
        population = sort_population(population)

        population = population[:POPULATION_SIZE]

        for solution in population:
            if best_solution is None or solution.distance < best_solution.distance and solution.vehicleNumber <= vehicleNumber:
                best_solution = solution

        # 打印第一个解
        print("[" + data + "]", "迭代次数：", str(n_iter + 1) + "/" + str(ITERATIONS), "车辆数量：", best_solution.vehicleNumber, "距离：", format(best_solution.distance, '.4f'), end='\t')

        # 编码
        population_code = encoder(population[:round(len(population) * 0.8)], len(customers) - 1)

        # 获取种群的概率分布
        population_score = get_population_probability_score(population[:round(len(population) * 0.8)])

        # 数据预处理
        ts_population_code = torch.tensor(population_code, dtype=torch.float32)
        ts_population_score = torch.tensor(population_score, dtype=torch.float32).squeeze(1)

        # 训练模型
        for epoch in range(epochs):
            # 前向传播
            outputs = model(ts_population_code)
            loss = criterion(outputs, ts_population_score)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印损失信息
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        # print("Training finished!")

        # 选择大于0.5的个体
        with torch.no_grad():
            predicted_scores = model(ts_population_code)
        selected_indices1 = torch.where(predicted_scores[:, 0] > 0.5)[0]
        ts_selected_population1 = ts_population_code[selected_indices1]
        selected_indices2 = torch.where(predicted_scores[:, 1] > 0.5)[0]
        ts_selected_population2 = ts_population_code[selected_indices2]
        mask = torch.logical_or(predicted_scores[:, 0] > 0.5, predicted_scores[:, 1] > 0.5)
        print("大于0.5的个体数量：", len(ts_population_code[torch.where(mask)[0]]), end=" ")

        selected_population_code1 = ts_selected_population1.numpy().tolist()
        selected_population_code2 = ts_selected_population2.numpy().tolist()
        selected_population1 = decoder(selected_population_code1, customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix)
        selected_population2 = decoder(selected_population_code2, customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix)

        # 产生新个体
        print("生成新个体...", end=" ")
        new_population = get_new_population_code(selected_population1, selected_population2, 1, 1, customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix)
        print("\t完成")

        # 添加新个体
        for solution in new_population:
            if checkUniqueness(population, solution):
                population.append(solution)

        if (n_iter + 1) % 20 == 0:
            print("vehicleNumber=" + str(best_solution.vehicleNumber) + "\n" + "distance=" + str(best_solution.distance))
            for route in best_solution.routes:
                print("\t" + "Route: 0 -> ", end="")
                for p in route.customers[1:]:
                    print(p.id, end="")
                    if p.id != route.customers[-1].id:
                        print(" --> ", end="")
                print("waitTime=" + str(route.waitTime))



    # # 解码
    # print("解码...", end=" ")
    # population = decoder(population_code[:POPULATION_SIZE], customers, VEHICLE_CAPACITY, MAX_DELAY_TIME, MAX_RETURN_TIME, distanceMatrix, timeMatrix)
    # population = calculate_strength(population)
    # population = calculate_r_fitness(population)
    # fitness_distanceMatrix = calculate_distance_matrix(population)
    # population = calculate_f_fitness(population, fitness_distanceMatrix, k)
    # population = sort_population(population)
    # print("\t完成")

    # 保存结果
    print("存储结果...", end=" ")
    if not os.path.exists("../results/solomon_100/" + data):
        os.makedirs("../results/solomon_100/" + data)
    with open("../results/solomon_100/" + data + "/population.pickle", "wb") as f:
        pickle.dump(population, f)
    with open("../results/solomon_100/" + data + "/best_solution.txt", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + data + "\n")
        f.write("vehicleNumber=" + str(best_solution.vehicleNumber) + "\n" + "distance=" + str(best_solution.distance) + "\n")
        for route in best_solution.routes:
            f.write("\t" + "Route: 0 -> ")
            for p in route.customers[1:]:
                f.write(str(p.id))
                if p.id != route.customers[-1].id:
                    f.write(" --> ")
            f.write("\twaitTime=" + str(route.waitTime) + "\n")
    print("\t完成")




# for p in population:
#     print("车辆数量：", p.vehicleNumber, "距离：", p.distance, "强度值：", p.strength, "原始适应度值：", p.r_fitness, "拥挤度适应度值：", p.f_fitness)


# # 非支配排序
# population, fronts = fast_non_dominated_sort(population)
# for idx, front in enumerate(fronts):
#     print(f"Front {idx+1}: {[population[i].fitness for i in front]} {[population[i].strength for i in front]} 原始适应度值 {[population[i].r_fitness for i in front]} 拥挤适应度值 {[population[i].f_fitness for i in front]}")
#
# # 计算拥挤度距离
# # distances = crowding_distance_assignment(fronts[0], population)
# for idx, front in enumerate(fronts):
#     print(f"\nFront {idx + 1}:")
#     distances = crowding_distance_assignment(front, population)
#     for j, sol_idx in enumerate(front):
#         print(f"Solution {sol_idx} crowding distance: {distances[j]}")
#
# # 选择


# # 输出初始种群中的第一个解
# print(population[0].vehicleNumber, population[0].distance, population[0].travelTime, population[0].waitTime, population[0].delayTime)
# for route in population[0].routes:
#     for p in route.customers:
#         print(p.id, end=" ")
#     print()
#     print("负载", route.load)
#     print("距离", route.distance)
#     print("旅行时间", route.travelTime)
#     print("等待时间", route.waitTime)
#     print("延误时间", route.delayTime)
#     print()
#
# for p in population:
#     print("车辆数量：", p.vehicleNumber, "距离：", p.distance, "旅行时间：", p.travelTime, "等待时间：", p.waitTime, "延误时间：", p.delayTime)



