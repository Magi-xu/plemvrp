from data_structures import *
import math


# 获取表示客户之间距离的矩阵
def getDistanceMatrix(customers):
    distanceMatrix = []
    for i in range(len(customers)):
        distanceMatrix.append([])
        for j in range(len(customers)):
            distanceMatrix[i].append(
                math.sqrt((customers[i].x - customers[j].x) ** 2 + (customers[i].y - customers[j].y) ** 2))
    return distanceMatrix


# 获取表示客户之间时间的矩阵
def getTimeMatrix(customers, distanceMatrix, vehicle_speed):
    timeMatrix = []
    for i in range(len(customers)):
        timeMatrix.append([])
        for j in range(len(customers)):
            timeMatrix[i].append(distanceMatrix[i][j] / vehicle_speed)
    return timeMatrix


# 读取数据
def read_vrptw_dataset(filename):
    data = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Read vehicle data
        data["name"] = lines[0].strip()
        vehicle_data = lines[4].split()
        data["vehicle_number"] = int(vehicle_data[0])
        data["vehicle_capacity"] = int(vehicle_data[1])

        # Read customer data
        data["customers"] = []
        for line in lines[9:]:
            customer_data = line.split()
            customer = {
                "cust_no": int(customer_data[0]),
                "x": int(customer_data[1]),
                "y": int(customer_data[2]),
                "demand": int(customer_data[3]),
                "ready_time": int(customer_data[4]),
                "due_date": int(customer_data[5]),
                "service_time": int(customer_data[6])
            }
            data["customers"].append(customer)

    # 创建客户对象列表
    customers = []
    for customer in data["customers"]:
        customers.append(
            Customer(customer["cust_no"], customer["x"], customer["y"], customer["demand"], customer["ready_time"],
                     customer["due_date"], customer["service_time"]))

    return data["name"], data["vehicle_number"], data["vehicle_capacity"], customers


# 约束条件检查
def checkConstraint(solution, vehicleCapacity, maxDelayTime, maxReturnTime):
    for route in solution.routes:
        if route.load > vehicleCapacity:
            return False
        for delayTime in route.delayTimeMatrix:
            if delayTime > maxDelayTime:
                return False
        if route.travelTime > maxReturnTime:
            return False
    return True


# 解决方案唯一性检查
def checkUniqueness(population, solution):
    for s in population:
        if s.distance == solution.distance and s.travelTime == solution.travelTime and s.waitTime == solution.waitTime and s.delayTime == solution.delayTime:
            return False
    return True


# 非支配排序
def fast_non_dominated_sort(population):
    def dominate(p, q):
        """Determine if solution p dominates solution q."""
        less_than = False
        for a, b in zip(p.fitness, q.fitness):
            if a > b:
                return False
            if a < b:
                less_than = True
        return less_than

    # Initialize all the solutions as rank 0 and empty dominated set
    ranks = [0] * len(population)
    dominated_solutions = [set() for _ in population]
    dominates_counts = [0] * len(population)

    # Calculate dominated solutions and count of solutions that dominate the current one
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j:
                continue
            if dominate(population[i], population[j]):
                dominated_solutions[i].add(j)
            elif dominate(population[j], population[i]):
                dominates_counts[i] += 1

    # Solutions with count 0 belong to the first front
    current_front = [i for i in range(len(population)) if dominates_counts[i] == 0]

    fronts = [current_front]

    # Extract subsequent fronts
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                dominates_counts[j] -= 1
                if dominates_counts[j] == 0:
                    next_front.append(j)
        current_front = next_front
        if current_front:
            fronts.append(current_front)

    return population, fronts


# 拥挤距离计算
def crowding_distance_assignment(front, population):
    n = len(front)
    if n == 0:
        return []

    # Initialize distance to 0 for all members of the front
    distances = [0.0] * n

    # Get the number of objectives
    num_objectives = len(population[front[0]].fitness)

    for m in range(num_objectives):
        # Sort the front using the m-th objective value
        sorted_front = sorted(front, key=lambda i: population[i].fitness[m])

        # Set the boundary points' distances to infinity
        # (assures they are always selected)
        distances[front.index(sorted_front[0])] = float('inf')
        distances[front.index(sorted_front[-1])] = float('inf')

        # Calculate the normalized distance (crowding distance) in the m-th objective
        objective_range = (population[sorted_front[-1]].fitness[m] - population[sorted_front[0]].fitness[m])
        for i in range(1, n - 1):
            if objective_range == 0:
                distances[front.index(sorted_front[i])] += 0
            else:
                distances[front.index(sorted_front[i])] += (population[sorted_front[i + 1]].fitness[m] - population[sorted_front[i - 1]].fitness[m]) / objective_range

    return distances


# 计算强度值
def calculate_strength(population):
    for p in population:
        for s in population:
            if s.fitness[0] > p.fitness[0] and s.fitness[1] > p.fitness[1]:
                p.strength += 1
    return population


# 计算原始适应度
def calculate_r_fitness(population):
    for p in population:
        for s in population:
            if s.fitness[0] < p.fitness[0] and s.fitness[1] < p.fitness[1]:
                p.r_fitness += s.strength
    return population


# 计算距离矩阵
def calculate_distance_matrix(population):
    def euclidean_distance(individual1, individual2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(individual1, individual2)))

    n = len(population)
    distances = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = euclidean_distance(population[i].fitness, population[j].fitness)
    return distances


# 计算计算拥挤度适应度
def calculate_f_fitness(population, fitness_distanceMatrix, k):
    # 计算kth距离
    kth_distances = []
    for i in range(len(population)):
        sorted_distances = sorted(fitness_distanceMatrix[i])
        kth_distance = sorted_distances[int(k)]
        kth_distances.append(kth_distance)

    # 计算适应度值
    for i in range(len(population)):
        density_estimate = 1.0 / (kth_distances[i] + 2)
        population[i].f_fitness = population[i].r_fitness + density_estimate

    return population


# 根据拥挤度适应度对种群进行排序
def sort_population(population):
    # 根据拥挤度适应度对种群进行排序，从小到大
    return sorted(population, key=lambda x: x.f_fitness)


def merge_rules(rules):
    is_fully_merged = True
    for round1 in rules:
        if round1[0] == round1[1]:
            rules.remove(round1)
            is_fully_merged = False
        else:
            for round2 in rules:
                if round2[0] == round1[1]:
                    rules.append((round1[0], round2[1]))
                    rules.remove(round1)
                    rules.remove(round2)
                    is_fully_merged = False
    return rules, is_fully_merged