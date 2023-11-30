import copy

from utils import *
import random


# 种群初始化
def init_population(population_size, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    population = []
    while len(population) < population_size:
        routes = []
        # 客户编号列表
        customer_ids = [i for i in range(1, len(customers))]
        # 随机打乱客户编号列表
        random.shuffle(customer_ids)
        route = [customers[0], customers[0]]
        routes.append(route)
        activate_route = 0
        while len(customer_ids) > 0:
            # 从客户编号列表中取出一个客户编号
            customer_id = customer_ids.pop()
            # 将该客户插入到路径的最后一个位置的前面
            routes[activate_route].insert(-1, customers[customer_id])
            # 检查约束条件
            if not checkConstraint(Solution([Route(routes[activate_route], distanceMatrix, timeMatrix)]),
                                   vehicle_capacity, maxDelayTime, maxReturnTime):
                # 将该客户从路径中删除
                routes[activate_route].pop(-2)
                # 在其他路径中查找是否存在满足约束条件的路径
                sf = False
                for n in range(len(routes)):
                    # 尝试插入到第一个位置到最后一个位置之间的任意位置
                    for j in range(1, len(routes[n])):
                        routes[n].insert(j, customers[customer_id])
                        if checkConstraint(Solution([Route(routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                           maxDelayTime, maxReturnTime):
                            activate_route = n
                            sf = True
                            break
                        else:
                            routes[n].pop(j)
                    if sf:
                        break
                # 如果没有找到满足约束条件的路径，则新建一条路径
                if not sf:
                    route = [customers[0], customers[0]]
                    routes.append(route)
                    activate_route = len(routes) - 1
                    routes[activate_route].insert(-1, customers[customer_id])
            else:
                # 如果满足约束条件，则继续检查下一个客户
                continue
        # 用routes创建解
        solution = Solution([Route(route, distanceMatrix, timeMatrix) for route in routes])
        # 检查唯一性
        if checkUniqueness(population, solution):
            population.append(solution)

    return population


# 随机PFIH初始化
def random_pfihs_init(population_size, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix,
                      timeMatrix):
    population = []
    while len(population) < population_size:
        # 将customers[1:]按照距离customers[0]的距离从大到小排序
        p_customers = sorted(customers[1:], key=lambda x: distanceMatrix[0][x.id], reverse=True)
        routes = []
        route = [customers[0], p_customers.pop(), customers[0]]
        routes.append(route)
        while len(p_customers) > 0:
            # 从p_customers中随机取出一个客户
            i = random.randint(0, len(p_customers) - 1)
            customer = p_customers.pop(i)
            # 将该客户尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受对distance产生最小影响的插入位置
            min_distance = float("inf")
            min_route = 0
            min_position = 0
            sf = False
            for n in range(len(routes)):
                for j in range(1, len(routes[n])):
                    routes[n].insert(j, customer)
                    if checkConstraint(Solution([Route(routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                       maxDelayTime, maxReturnTime):
                        distance = Route(routes[n], distanceMatrix, timeMatrix).distance
                        if distance < min_distance:
                            min_distance = distance
                            min_route = n
                            min_position = j
                            sf = True
                        routes[n].pop(j)
                    else:
                        routes[n].pop(j)
            if sf:
                # 将该客户插入到对distance产生最小影响的路径的对应位置
                routes[min_route].insert(min_position, customer)
            else:
                # 如果没有找到满足约束条件的路径
                # 将customer放回到原位置
                p_customers.insert(i, customer)
                # #则新建一条路径
                route = [customers[0], p_customers.pop(), customers[0]]
                routes.append(route)
        # 用routes创建解
        solution = Solution([Route(route, distanceMatrix, timeMatrix) for route in routes])
        # 检查唯一性
        if checkUniqueness(population, solution):
            population.append(solution)

    return population


# 编码器
def encoder(population, n):
    population_code = []
    for solution in population:
        code = [0 for i in range(n)]
        priority = n
        for route in solution.routes:
            for i in range(len(route.customers) - 2):
                code[route.customers[i + 1].id - 1] = priority
                priority -= 1
        population_code.append(code)
    return population_code


# 解码器
def decoder(population_code, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    population = []
    for code in population_code:
        n = len(code)
        max_priority = n
        routes = []
        route = [customers[0], customers[0]]
        routes.append(route)
        activate_route = 0
        # 当code不全为0时
        while code.count(0) != n:
            customer_id = code.index(max_priority)
            customer = customers[customer_id + 1]
            routes[activate_route].insert(-1, customer)
            if not checkConstraint(Solution([Route(routes[activate_route], distanceMatrix, timeMatrix)]),
                                   vehicle_capacity, maxDelayTime, maxReturnTime):
                routes[activate_route].pop(-2)
                routes.append([customers[0], customer, customers[0]])
                activate_route += 1
            code[customer_id] = 0
            max_priority -= 1
        for route in routes:
            if len(route) == 2:
                routes.remove(route)
        if len(routes) > 0:
            solution = Solution([Route(route, distanceMatrix, timeMatrix) for route in routes])
            population.append(solution)
    return population


# 获取种群概率分数
def get_population_probability_score(population):
    p1 = copy.copy(population)
    p2 = copy.copy(population)
    # p1按照车辆数从小到大排序
    p1.sort(key=lambda x: x.vehicleNumber)
    # p2按照距离从小到大排序
    p2.sort(key=lambda x: x.distance)
    score1 = 1
    score2 = 1
    probability_score = [[0, 0] for i in range(len(population))]
    for p in p1:
        # 找到p在population中的索引
        p_index = population.index(p)
        probability_score[p_index][0] = score1
        score1 -= 1 / len(population)
    for p in p2:
        # 找到p在population中的索引
        p_index = population.index(p)
        probability_score[p_index][1] = score2
        score2 -= 1 / len(population)
    return probability_score

    # probability_score = []
    # s = 1
    # for solution in population:
    #     probability_score.append(s)
    #     s -= 1 / len(population)
    # return probability_score


# Function SequentialConstructiveCrossover(parent1, parent2):
#     Initialize offspring to an empty route
#     Initialize a set of unvisited customers

#     While there are unvisited customers:
#         Select the current customer from parent1 or parent2 (start with the first customer of parent1)

#         If the current customer is already in offspring:
#             Switch to the other parent and select the next customer
#             Continue

#         Determine the feasible time window for the current customer based on the offspring route
#         If adding the current customer violates constraints (e.g., vehicle capacity or time window):
#             Find the next available customer in the same parent that doesn't violate constraints
#             If not found, switch to the other parent and continue searching

#         Add the selected customer to the offspring route
#         Remove the selected customer from the set of unvisited customers

#     Return offspring


# def cx_partially_matched(ind1, ind2):
#     cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
#     part1 = ind2[cxpoint1:cxpoint2 + 1]
#     part2 = ind1[cxpoint1:cxpoint2 + 1]
#     rule1to2 = list(zip(part1, part2))
#     is_fully_merged = False
#     while not is_fully_merged:
#         rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
#     rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
#     rule1to2 = dict(rule1to2)
#     ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + \
#            [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2 + 1:]]
#     ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + \
#            [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2 + 1:]]
#     return ind1, ind2


# def fill_remaining(chromosome: list, filling: list) -> list:
#     fill_index = 0
#     for ch_index, ch_bit in enumerate(chromosome):
#         if ch_bit is None:
#             while filling[fill_index] in chromosome:
#                 fill_index += 1
#             chromosome[ch_index] = filling[fill_index]
#
#     return chromosome
#
#
# def crossover_uox(parent1: list, parent2: list) -> (list, list):
#     if len(parent1) != len(parent2):
#         raise Exception("Crossover error: Parents length are not equal")
#     chrome_len = len(parent1)
#     mask_binary = np.random.randint(2, size=chrome_len)
#     # mask_binary = [0, 1, 1, 0, 1, 1]
#     child1 = []
#     child2 = []
#
#     for index, mask in enumerate(mask_binary):
#         if mask:
#             child1.append(parent1[index])
#             child2.append(parent2[index])
#         else:
#             child1.append(None)
#             child2.append(None)
#
#     child1 = fill_remaining(child1, parent2)
#     child2 = fill_remaining(child2, parent1)
#
#     return child1, child2
#
#
# def crossover_cx(parent1: list, parent2: list) -> (list, list):
#     length = len(parent1)
#     if length != len(parent2):
#         raise Exception("Crossover error: Parents length are not equal")
#
#     p1 = {}
#     p2 = {}
#     p1_inv = {}
#     p2_inv = {}
#     for i in range(length):
#         p1[i] = parent1[i]
#         p1_inv[p1[i]] = i
#         p2[i] = parent2[i]
#         p2_inv[p2[i]] = i
#
#     cycles_indices = []
#     while p1 != {}:
#         i = min(list(p1.keys()))
#         cycle = [i]
#         start = p1[i]
#         check = p2[i]
#         del p1[i]
#         del p2[i]
#
#         while check != start:
#             i = p1_inv[check]
#             cycle.append(i)
#             check = p2[i]
#             del p1[i]
#             del p2[i]
#
#         cycles_indices.append(cycle)
#
#     child = ({}, {})
#
#     for run, indices in enumerate(cycles_indices):
#         first = run % 2
#         second = (first + 1) % 2
#
#         for i in indices:
#             child[first][i] = parent1[i]
#             child[second][i] = parent2[i]
#
#     child1 = []
#     child2 = []
#     for i in range(length):
#         child1.append(child[0][i])
#         child2.append(child[1][i])
#
#     return child1, child2
#
#     # child1, child2 = crossover_cx([8, 4, 7, 3, 6, 2, 5, 1, 9, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#     # print(child1)   # 8 1 2 3 4 5 6 7 9 0
#     # print(child2)   # 0 4 7 3 6 2 5 1 8 9
#
#
# def crossover_pmx(parent1: list, parent2: list) -> (list, list):
#     if len(parent1) != len(parent2):
#         raise Exception("Crossover error: Parents length are not equal")
#     chrome_len = len(parent1)
#     p1_inv = {}
#     p2_inv = {}
#     child1 = list(parent1)
#     child2 = list(parent2)
#
#     for i in range(chrome_len):
#         p1_inv[parent1[i]] = i
#         p2_inv[parent2[i]] = i
#
#     # crossover points random from 0 to chrome_len
#     cx_point1 = np.random.randint(0, chrome_len)
#     cx_point2 = np.random.randint(0, chrome_len - 1)
#
#     # prepare it for loop from point1 to point2
#     if cx_point2 >= cx_point1:
#         cx_point2 += 1
#     else:
#         cx_point1, cx_point2 = cx_point2, cx_point1
#
#     # Loop between crossover points
#     for i in range(cx_point1, cx_point2):
#         # Selected values before changing the parent
#         check1 = child1[i]
#         check2 = child2[i]
#         # Swap matched !
#         parent1[i], parent1[p1_inv[check2]] = check2, check1
#         parent2[i], parent2[p2_inv[check1]] = check1, check2
#
#         p1_inv[check1] = p1_inv[check2]
#         p1_inv[check2] = p1_inv[check1]
#         p2_inv[check1] = p2_inv[check2]
#         p2_inv[check2] = p2_inv[check1]
#
#     return child1, child2


# def mut_inverse_indexes(individual):
#     start, stop = sorted(random.sample(range(len(individual)), 2))
#     temp = individual[start:stop + 1]
#     temp.reverse()
#     individual[start:stop + 1] = temp
#     return individual


# def mutation_slice_points(chromosome_len: int) -> (int, int):
#     slice_point_1 = random.randint(0, chromosome_len - 3)
#     slice_point_2 = random.randint(slice_point_1 + 2, chromosome_len - 1)
#     return slice_point_1, slice_point_2
#
#
# def mutation_inversion(chromosome: list) -> list:
#     slice_point_1, slice_point_2 = mutation_slice_points(len(chromosome))
#
#     return chromosome[:slice_point_1] + list(reversed(chromosome[slice_point_1:slice_point_2])) + chromosome[slice_point_2:]
#
#
# def mutation_scramble(chromosome: list) -> list:
#     slice_point_1, slice_point_2 = mutation_slice_points(len(chromosome))
#     scrambled = chromosome[slice_point_1:slice_point_2]
#     random.shuffle(scrambled)
#
#     return chromosome[:slice_point_1] + scrambled + chromosome[slice_point_2:]
#
#
# def cross(parent1, parent2):
#     n_customer = len(parent1)
#     if parent1 != parent2:
#         cro1_index = int(random.randint(0, n_customer - 1))
#         cro2_index = int(random.randint(cro1_index, n_customer - 1))
#         new_c1_f = []
#         new_c1_m = parent1[cro1_index:cro2_index + 1]
#         new_c1_b = []
#         new_c2_f = []
#         new_c2_m = parent2[cro1_index:cro2_index + 1]
#         new_c2_b = []
#         for index in range(n_customer):
#             if len(new_c1_f) < cro1_index:
#                 if parent2[index] not in new_c1_m:
#                     new_c1_f.append(parent2[index])
#             else:
#                 if parent2[index] not in new_c1_m:
#                     new_c1_b.append(parent2[index])
#         for index in range(n_customer):
#             if len(new_c2_f) < cro1_index:
#                 if parent1[index] not in new_c2_m:
#                     new_c2_f.append(parent1[index])
#             else:
#                 if parent1[index] not in new_c2_m:
#                     new_c2_b.append(parent1[index])
#         new_c1 = new_c1_f + new_c1_m + new_c1_b
#         new_c2 = new_c2_f + new_c2_m + new_c2_b
#     else:
#         new_c1 = parent1
#         new_c2 = parent2
#     return new_c1, new_c2
#
#
# def mutation(individual):
#     n_customers = len(individual)
#     f1 = individual
#     m1_index = random.randint(0, n_customers - 1)
#     m2_index = random.randint(0, n_customers - 1)
#     if m1_index != m2_index:
#         node1 = f1[m1_index]
#         f1[m1_index] = f1[m2_index]
#         f1[m2_index] = node1
#     return f1


# 为了减少车辆数量，k次迭代选出两个父代中的客户数最多的路线，剩下的客户尝试插入
def cross1(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    # 复制parent1, parent2
    c_parent1 = copy.deepcopy(parent1)
    c_parent2 = copy.deepcopy(parent2)
    k = min(len(parent1.routes), len(parent2.routes)) - 1
    parents = [c_parent1, c_parent2]
    child_routes = []
    for i in range(k):
        f_route = c_parent1.routes[0]
        parent_index = 0
        for route in c_parent1.routes:
            if len(route.customers) > len(f_route.customers):
                f_route = route
                parent_index = 0
        for route in c_parent2.routes:
            if len(route.customers) > len(f_route.customers):
                f_route = route
                parent_index = 1
        child_routes.append([x for x in f_route.customers])
        parents[parent_index].routes.remove(f_route)
        # 在另一个父代中查找与f_route中的客户并删除
        parent_index = 1 - parent_index
        for customer in f_route.customers[1:-1]:
            for route in parents[parent_index].routes:
                if customer in route.customers:
                    route.customers.remove(customer)
        # 删除空路径
        for route in parents[parent_index].routes:
            if len(route.customers) == 2:
                parents[parent_index].routes.remove(route)

    unvisited_customers = []
    for route in parents[0].routes + parents[1].routes:
        for customer in route.customers[1:-1]:
            unvisited_customers.append(customer)
    while len(unvisited_customers) > 0:
        # 随机选出一个客户
        customer = unvisited_customers.pop(random.randint(0, len(unvisited_customers) - 1))
        # 尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受第一个满足约束的位置
        route_index = 0
        position_index = 0
        sf = False
        for n in range(len(child_routes)):
            for j in range(1, len(child_routes[n])):
                child_routes[n].insert(j, customer)
                if checkConstraint(Solution([Route(child_routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                   maxDelayTime, maxReturnTime):
                    route_index = n
                    position_index = j
                    sf = True
                    child_routes[n].pop(j)
                    break
                else:
                    child_routes[n].pop(j)
        if sf:
            # 将该客户插入到对应位置
            child_routes[route_index].insert(position_index, customer)
        else:
            # 如果没有找到满足约束条件的路径 则新建一条路径
            route = [customers[0], customer, customers[0]]
            child_routes.append(route)

    # 用child_routes创建解
    child = Solution([Route(route, distanceMatrix, timeMatrix) for route in child_routes])
    return child


# 为了减少行驶距离，k次迭代选出两个父代中行驶距离与路线中乘客数量之比最小的路线，剩下的客户尝试插入
def cross2(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    # 复制parent1, parent2
    c_parent1 = copy.deepcopy(parent1)
    c_parent2 = copy.deepcopy(parent2)
    k = min(len(parent1.routes), len(parent2.routes))
    parents = [c_parent1, c_parent2]
    child_routes = []
    for i in range(k):
        f_route = c_parent1.routes[0]
        parent_index = 0
        for route in c_parent1.routes:
            if route.distance / len(route.customers) < f_route.distance / len(f_route.customers):
                f_route = route
                parent_index = 0
        for route in c_parent2.routes:
            if route.distance / len(route.customers) < f_route.distance / len(f_route.customers):
                f_route = route
                parent_index = 1
        child_routes.append([x for x in f_route.customers])
        parents[parent_index].routes.remove(f_route)
        # 在另一个父代中查找与f_route中的客户并删除
        parent_index = 1 - parent_index
        for customer in f_route.customers[1:-1]:
            for route in parents[parent_index].routes:
                if customer in route.customers:
                    route.customers.remove(customer)
        # 删除空路径
        for route in parents[parent_index].routes:
            if len(route.customers) == 2:
                parents[parent_index].routes.remove(route)

    unvisited_customers = []
    for route in parents[0].routes + parents[1].routes:
        for customer in route.customers[1:-1]:
            unvisited_customers.append(customer)
    while len(unvisited_customers) > 0:
        # 随机选出一个客户
        customer = unvisited_customers.pop(random.randint(0, len(unvisited_customers) - 1))
        # 尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受对distance产生最小影响的插入位置
        min_distance = float("inf")
        min_route_index = 0
        min_position_index = 0
        sf = False
        for n in range(len(child_routes)):
            for j in range(1, len(child_routes[n])):
                child_routes[n].insert(j, customer)
                if checkConstraint(Solution([Route(child_routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                   maxDelayTime, maxReturnTime):
                    distance = Route(child_routes[n], distanceMatrix, timeMatrix).distance
                    if distance < min_distance:
                        min_distance = distance
                        min_route_index = n
                        min_position_index = j
                        sf = True
                    child_routes[n].pop(j)
                else:
                    child_routes[n].pop(j)
        if sf:
            # 将该客户插入到对distance产生最小影响的路径的对应位置
            child_routes[min_route_index].insert(min_position_index, customer)
        else:
            # 如果没有找到满足约束条件的路径 则新建一条路径
            route = [customers[0], customer, customers[0]]
            child_routes.append(route)

    # 用child_routes创建解
    child = Solution([Route(route, distanceMatrix, timeMatrix) for route in child_routes])
    return child


def mutation1(individual, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    # 提取individual中各路径的客户列表
    new_routes = []
    for route in individual.routes:
        new_routes.append([x for x in route.customers])
    select_route = new_routes[0]
    new_routes_bak = copy.deepcopy(new_routes)
    while True:
        select_route_index = 0
        for route in new_routes:
            if len(route) < len(select_route):
                select_route_index = new_routes.index(route)
        select_route = new_routes[select_route_index]
        if len(select_route) == 2:
            break
        new_routes.remove(select_route)
        unvisited_customers = select_route[1:-1]
        no_position = False
        while len(unvisited_customers) > 0:
            # 随机选出一个客户
            customer = unvisited_customers.pop(random.randint(0, len(unvisited_customers) - 1))
            # 尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受第一个满足约束的位置
            route_index = 0
            position_index = 0
            sf = False
            for n in range(len(new_routes)):
                for j in range(1, len(new_routes[n])):
                    new_routes[n].insert(j, customer)
                    if checkConstraint(Solution([Route(new_routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                       maxDelayTime, maxReturnTime):
                        route_index = n
                        position_index = j
                        sf = True
                        new_routes[n].pop(j)
                        break
                    else:
                        new_routes[n].pop(j)
            if sf:
                # 将该客户插入到对应位置
                new_routes[route_index].insert(position_index, customer)
            else:
                # 如果没有找到满足约束条件的路径 则放弃变异
                no_position = True
                break
        if no_position:
            new_routes = copy.deepcopy(new_routes_bak)
            break
        else:
            new_routes_bak = copy.deepcopy(new_routes)
    # 用new_routes创建解
    new_individual = Solution([Route(route, distanceMatrix, timeMatrix) for route in new_routes])
    return new_individual


# 取最短的两条路线，尝试插入合并
def mutation11(individual, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    # 提取individual中各路径的客户列表
    new_routes = []
    for route in individual.routes:
        new_routes.append([x for x in route.customers])
    select_route_index = 0
    for route in new_routes:
        if len(route) < len(new_routes[select_route_index]):
            select_route_index = new_routes.index(route)
    select_route = new_routes[select_route_index]
    new_routes.remove(select_route)
    unvisited_customers = select_route[1:-1]
    select_route_index = 0
    for route in new_routes:
        if len(route) < len(new_routes[select_route_index]):
            select_route_index = new_routes.index(route)
    select_route = new_routes[select_route_index]
    new_routes.remove(select_route)
    unvisited_customers.extend(select_route[1:-1])
    while len(unvisited_customers) > 0:
        # 随机选出一个客户
        customer = unvisited_customers.pop(random.randint(0, len(unvisited_customers) - 1))
        # 尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受第一个满足约束的位置
        route_index = 0
        position_index = 0
        sf = False
        for n in range(len(new_routes)):
            for j in range(1, len(new_routes[n])):
                new_routes[n].insert(j, customer)
                if checkConstraint(Solution([Route(new_routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                   maxDelayTime, maxReturnTime):
                    route_index = n
                    position_index = j
                    sf = True
                    new_routes[n].pop(j)
                    break
                else:
                    new_routes[n].pop(j)
        if sf:
            # 将该客户插入到对应位置
            new_routes[route_index].insert(position_index, customer)
        else:
            # 如果没有找到满足约束条件的路径 则新建一条路径
            route = [new_routes[0][0], customer, new_routes[0][0]]
            new_routes.append(route)

    # 用new_routes创建解
    new_individual = Solution([Route(route, distanceMatrix, timeMatrix) for route in new_routes])
    return new_individual


def mutation2(individual, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    # 提取individual中各路径的客户列表
    new_routes = []
    for route in individual.routes:
        new_routes.append([x for x in route.customers])
    unvisited_customers = []
    v = 4
    if random.random() < 0.7:
        # 随机选择一条路
        select_route_index = random.randint(0, len(new_routes) - 1)
        # 取出随机个数的客户
        for i in range(random.randint(1, len(new_routes[select_route_index]) - 2)):
            customer = new_routes[select_route_index].pop(random.randint(1, len(new_routes[select_route_index]) - 2))
            unvisited_customers.append(customer)
            if len(new_routes[select_route_index]) == 2:
                new_routes.remove(new_routes[select_route_index])
    else:
        for i in range(v):
            select_route_index = random.randint(0, len(new_routes) - 1)
            customer = new_routes[select_route_index].pop(random.randint(1, len(new_routes[select_route_index]) - 2))
            unvisited_customers.append(customer)
            if len(new_routes[select_route_index]) == 2:
                new_routes.remove(new_routes[select_route_index])

    while len(unvisited_customers) > 0:
        # 随机选出一个客户
        customer = unvisited_customers.pop(random.randint(0, len(unvisited_customers) - 1))
        # 尝试插入到每一条路径中的第一个位置到最后一个位置之间的任意位置，接受对distance产生最小影响的插入位置
        min_distance = float("inf")
        min_route_index = 0
        min_position_index = 0
        sf = False
        for n in range(len(new_routes)):
            for j in range(1, len(new_routes[n])):
                new_routes[n].insert(j, customer)
                if checkConstraint(Solution([Route(new_routes[n], distanceMatrix, timeMatrix)]), vehicle_capacity,
                                   maxDelayTime, maxReturnTime):
                    distance = Route(new_routes[n], distanceMatrix, timeMatrix).distance
                    if distance < min_distance:
                        min_distance = distance
                        min_route_index = n
                        min_position_index = j
                        sf = True
                    new_routes[n].pop(j)
                else:
                    new_routes[n].pop(j)
        if sf:
            # 将该客户插入到对distance产生最小影响的路径的对应位置
            new_routes[min_route_index].insert(min_position_index, customer)
        else:
            # 如果没有找到满足约束条件的路径 则新建一条路径
            route = [customers[0], customer, customers[0]]
            new_routes.append(route)

    # 用new_routes创建解
    new_individual = Solution([Route(route, distanceMatrix, timeMatrix) for route in new_routes])
    return new_individual


#
def get_new_population_code(population1, population2, cx_pb, mut_pb, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix):
    new_population = []
    # 交叉
    for parent1, parent2 in zip(population1[::2], population1[1::2]):
        if random.random() < cx_pb:
            # 交叉
            child1 = cross1(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # child2 = cross2(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # 添加新个体
            new_population.append(child1)
            # new_population.append(child2)
    for mutant in population1:
        if random.random() < mut_pb:
            # 变异
            # mutant1 = mutation1(mutant, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            mutant11 = mutation11(mutant, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # mutant2 = mutation2(mutant, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # 添加新个体
            # new_population.append(mutant1)
            new_population.append(mutant11)
            # new_population.append(mutant2)
    for parent1, parent2 in zip(population2[::2], population2[1::2]):
        if random.random() < cx_pb:
            # 交叉
            # child1 = cross1(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            child2 = cross2(parent1, parent2, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # 添加新个体
            # new_population.append(child1)
            new_population.append(child2)
    for mutant in population2:
        if random.random() < mut_pb:
            # 变异
            # mutant1 = mutation1(mutant, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            mutant2 = mutation2(mutant, customers, vehicle_capacity, maxDelayTime, maxReturnTime, distanceMatrix, timeMatrix)
            # 添加新个体
            # new_population.append(mutant1)
            new_population.append(mutant2)
    return new_population
