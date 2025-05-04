import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def read_tsp_file(filename):
    """读取TSP文件并返回城市坐标列表"""
    with open(filename, "r") as f:
        lines = f.readlines()

    # 找到坐标数据开始的标记行
    start = 0
    while not lines[start].strip().startswith("NODE_COORD_SECTION"):
        start += 1

    # 读取坐标数据
    coords = []
    for line in lines[start + 1 :]:
        if line.strip() == "EOF":  # 文件结束标记
            break
        parts = line.strip().split()
        coords.append((float(parts[1]), float(parts[2])))  # 提取x,y坐标

    return np.array(coords)


class TabuSearchTSP:
    """禁忌搜索算法解决TSP问题"""

    def __init__(
        self,
        coords,
        tabu_size=50,  # 禁忌表大小
        max_iter=2000,  # 最大迭代次数
        neighbor_size=100,  # 邻域大小
        diversification_freq=100,  # 多样化频率
        aspiration_ratio=1.05,  # 渴望准则比率
        k_nearest=10,  # 考虑的近邻数量
    ):
        """初始化禁忌搜索参数"""
        self.coords = coords
        self.num_cities = len(coords)
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.neighbor_size = neighbor_size
        self.diversification_freq = diversification_freq
        self.aspiration_ratio = aspiration_ratio
        self.k_nearest = k_nearest

        # 预计算距离矩阵
        x = coords[:, 0]
        y = coords[:, 1]
        self.distance_matrix = np.sqrt(
            (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        )

        # 为每个城市预计算k近邻
        self.nearest_neighbors = np.argsort(self.distance_matrix, axis=1)[
            :, 1 : k_nearest + 1
        ]

        # 禁忌表实现
        self.tabu_list = defaultdict(int)
        # 移动频率统计
        self.move_frequency = defaultdict(int)

    def calculate_distance(self, route):
        """精确计算TSP路径的距离"""
        total = 0.0
        n = len(route)
        for i in range(n):
            total += self.distance_matrix[route[i]][route[(i + 1) % n]]
        return total

    def generate_initial_solution(self):
        """使用最近邻启发式生成初始解"""
        best_route = None
        best_dist = float("inf")

        # 从5个不同的起点尝试，选择最好的一个
        for _ in range(5):
            start = random.randint(0, self.num_cities - 1)
            route = [start]
            unvisited = set(range(self.num_cities))
            unvisited.remove(start)

            while unvisited:
                last = route[-1]
                # 从预计算的列表中获取未访问的最近城市
                candidates = [n for n in self.nearest_neighbors[last] if n in unvisited]
                if not candidates:  # 如果没有候选城市，从剩余未访问城市中选择
                    candidates = list(unvisited)

                # 选择距离最近的城市
                next_city = min(candidates, key=lambda x: self.distance_matrix[last, x])
                route.append(next_city)
                unvisited.remove(next_city)

            # 评估当前路径
            dist = self.calculate_distance(route)
            if dist < best_dist:
                best_dist = dist
                best_route = route

        return best_route

    def get_neighbors(self, route):
        """使用2-opt移动生成邻域解"""
        neighbors = []
        n = len(route)

        # 生成2-opt邻居
        for _ in range(self.neighbor_size):
            # 随机选择两个位置i和j(限制j在i的k_nearest范围内)
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, min(i + self.k_nearest + 1, n - 1))

            # 应用2-opt交换
            new_route = route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 :]

            # 计算精确距离
            new_dist = self.calculate_distance(new_route)

            # 记录移动类型和位置
            neighbors.append((new_route, ("2opt", i, j), new_dist))

        return neighbors

    def is_tabu(self, move, current_dist, best_dist):
        """检查移动是否被禁忌，考虑渴望准则"""
        if move in self.tabu_list:
            # 渴望准则: 如果新解足够好，可以突破禁忌
            return current_dist > best_dist * self.aspiration_ratio
        return False

    def update_tabu_list(self, move, iteration):
        """使用动态任期更新禁忌表"""
        # 基于移动频率的动态禁忌任期
        tenure = random.randint(5, 10) + self.move_frequency[move] // 3
        self.tabu_list[move] = iteration + tenure
        self.move_frequency[move] += 1

        # 定期清理过期的禁忌移动
        if iteration % 50 == 0:
            self.tabu_list = {m: t for m, t in self.tabu_list.items() if t > iteration}
            if len(self.tabu_list) > self.tabu_size:
                # 移除最旧的禁忌移动
                oldest = sorted(self.tabu_list.items(), key=lambda x: x[1])[
                    : len(self.tabu_list) - self.tabu_size
                ]
                for m, _ in oldest:
                    del self.tabu_list[m]

    def diversify_solution(self, current_route):
        """使用双桥移动进行受控多样化"""
        n = len(current_route)
        if n < 8:  # 如果城市太少，不进行多样化
            return current_route.copy()

        # 选择4个随机位置
        a, b, c, d = sorted(random.sample(range(n), 4))

        # 应用双桥移动
        new_route = (
            current_route[: a + 1]
            + current_route[d:]
            + current_route[b + 1 : c + 1]
            + current_route[a + 1 : b + 1]
            + current_route[c + 1 : d]
        )

        return new_route

    def solve(self):
        """主禁忌搜索算法"""
        # 生成初始解
        current_route = self.generate_initial_solution()
        current_dist = self.calculate_distance(current_route)
        best_route = current_route.copy()  # type: ignore
        best_dist = current_dist

        # 用于跟踪进度
        last_improvement = 0
        history = []

        for iteration in range(self.max_iter):
            # 检查停滞情况
            if iteration - last_improvement > self.diversification_freq:
                current_route = self.diversify_solution(current_route)
                current_dist = self.calculate_distance(current_route)
                print(f"在迭代 {iteration} 进行多样化")
                last_improvement = iteration

            # 生成邻居
            neighbors = self.get_neighbors(current_route)
            if not neighbors:
                continue

            # 找到最佳非禁忌邻居
            best_neighbor = None
            best_neighbor_dist = float("inf")
            best_move = None

            for neighbor, move, dist in neighbors:
                if (not self.is_tabu(move, dist, best_dist)) and (
                    dist < best_neighbor_dist
                ):
                    best_neighbor = neighbor
                    best_neighbor_dist = dist
                    best_move = move

            # 更新当前解
            if best_neighbor is not None:
                current_route = best_neighbor
                current_dist = best_neighbor_dist

                # 更新禁忌表
                if best_move is not None:
                    self.update_tabu_list(best_move, iteration)

                # 更新全局最佳
                if current_dist < best_dist:
                    best_route = current_route.copy()
                    best_dist = current_dist
                    last_improvement = iteration
                    print(f"迭代 {iteration}: 新的最佳距离: {best_dist:.2f}")

            # 记录历史
            history.append(best_dist)

            # 进度输出
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 当前最佳: {best_dist:.2f}")

        # 最终局部搜索
        print("运行最终的2-opt局部搜索...")
        improved = True
        while improved:
            improved = False
            neighbors = self.get_neighbors(best_route)
            for neighbor, _, dist in neighbors:
                if dist < best_dist:
                    best_route = neighbor
                    best_dist = dist
                    improved = True
                    print(f"最终改进: {best_dist:.2f}")
                    break

        return best_route, best_dist, history


class SimulatedAnnealingTSP:
    """模拟退火算法解决TSP问题"""

    def __init__(
        self,
        coords,
        initial_temp=10000,  # 初始温度
        cooling_rate=0.003,  # 冷却速率
        max_iter=2000,  # 最大迭代次数
        k_nearest=10,  # 考虑的近邻数量
    ):
        """初始化模拟退火参数"""
        self.coords = coords
        self.num_cities = len(coords)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.k_nearest = k_nearest

        # 预计算距离矩阵
        x = coords[:, 0]
        y = coords[:, 1]
        self.distance_matrix = np.sqrt(
            (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        )

        # 为每个城市预计算k近邻
        self.nearest_neighbors = np.argsort(self.distance_matrix, axis=1)[
            :, 1 : k_nearest + 1
        ]

    def calculate_distance(self, route):
        """计算TSP路径的距离"""
        total = 0.0
        n = len(route)
        for i in range(n):
            total += self.distance_matrix[route[i]][route[(i + 1) % n]]
        return total

    def generate_initial_solution(self):
        """生成随机初始解"""
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route

    def get_neighbor(self, route):
        """生成邻域解(使用2-opt移动)"""
        n = len(route)
        i = random.randint(0, n - 2)
        # 限制j在i的k近邻范围内
        j = random.randint(i + 1, min(i + self.k_nearest + 1, n - 1))

        # 应用2-opt交换
        new_route = route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 :]
        return new_route

    def solve(self):
        """主模拟退火算法"""
        # 生成初始解
        current_route = self.generate_initial_solution()
        current_dist = self.calculate_distance(current_route)
        best_route = current_route.copy()
        best_dist = current_dist

        # 初始化温度
        temp = self.initial_temp

        # 用于跟踪进度
        history = []

        for iteration in range(self.max_iter):
            # 生成邻域解
            new_route = self.get_neighbor(current_route)
            new_dist = self.calculate_distance(new_route)

            # 计算能量差
            delta = new_dist - current_dist

            # 决定是否接受新解
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_route = new_route
                current_dist = new_dist

                # 更新全局最佳
                if current_dist < best_dist:
                    best_route = current_route.copy()
                    best_dist = current_dist
                    print(f"迭代 {iteration}: 新的最佳距离: {best_dist:.2f}")

            # 降低温度
            temp *= 1 - self.cooling_rate

            # 记录历史
            history.append(best_dist)

            # 进度输出
            if iteration % 100 == 0:
                print(
                    f"迭代 {iteration}, 当前温度: {temp:.2f}, 当前最佳: {best_dist:.2f}"
                )

        # 最终局部搜索
        print("运行最终的2-opt局部搜索...")
        improved = True
        while improved:
            improved = False
            for i in range(1, self.num_cities - 1):
                for j in range(i + 1, min(i + self.k_nearest + 1, self.num_cities)):
                    # 尝试2-opt交换
                    new_route = (
                        best_route[:i]
                        + best_route[i : j + 1][::-1]
                        + best_route[j + 1 :]
                    )
                    new_dist = self.calculate_distance(new_route)
                    if new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        print(f"最终改进: {best_dist:.2f}")
                        break
                if improved:
                    break

        return best_route, best_dist, history


def plot_route(coords, route, title="TSP Route", save_path="tsp_route.png"):
    """绘制TSP路径图并保存为文件，显示城市序号"""
    plt.figure(figsize=(12, 8))
    route_coords = coords[np.array(route + [route[0]])]

    # 绘制城市点和路径线
    plt.scatter(coords[:, 0], coords[:, 1], c="red", s=100, edgecolors="black")
    plt.plot(route_coords[:, 0], route_coords[:, 1], "b-", linewidth=1.5, alpha=0.7)

    # 标记起点/终点
    start_point = coords[route[0]]
    plt.scatter(
        start_point[0], start_point[1], c="green", s=200, marker="*", edgecolors="black"
    )

    # 为每个城市添加序号标签
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=8, ha="center", va="center", color="white")

    # 为路径上的城市添加顺序编号
    for order, city_idx in enumerate(route):
        x, y = coords[city_idx]
        offset = 1.0  # 避免与城市序号重叠
        plt.text(
            x + offset,
            y + offset,
            str(order + 1),
            fontsize=8,
            ha="center",
            va="center",
            color="blue",
        )

    plt.title(f"{title}\nDistance: {calculate_route_distance(coords, route):.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 提高DPI使文字更清晰
    plt.close()


def calculate_route_distance(coords, route):
    """计算TSP路径的总距离"""
    route_coords = coords[np.array(route)]
    dx = np.diff(np.append(route_coords[:, 0], route_coords[0, 0]))
    dy = np.diff(np.append(route_coords[:, 1], route_coords[0, 1]))
    return np.sum(np.sqrt(dx**2 + dy**2))


def plot_convergence(history, title="Convergence History", save_path="convergence.png"):
    """绘制收敛历史图并保存为文件"""
    plt.figure(figsize=(12, 6))
    plt.plot(history, "b-", linewidth=1.5, alpha=0.7, label="Best Distance")

    # 标记改进点
    improvements = [0] + [
        i for i in range(1, len(history)) if history[i] < history[i - 1]
    ]
    if improvements:
        plt.scatter(
            improvements,
            [history[i] for i in improvements],
            c="red",
            s=50,
            zorder=5,
            label="Improvements",
        )

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def verify_route(coords, route):
    """验证路径是否有效"""
    if len(set(route)) != len(coords):
        print("错误: 某些城市缺失或重复!")
        return False

    # 验证距离计算一致性
    dist1 = calculate_route_distance(coords, route)
    dist2 = TabuSearchTSP(coords).calculate_distance(route)

    if not np.isclose(dist1, dist2):
        print(f"距离计算不匹配: {dist1} vs {dist2}")
        return False

    return True


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 读取TSP数据文件
    filename = "eil51.tsp"
    coords = read_tsp_file(filename)

    # 设置随机种子以保证可重复性
    seed = 262652  # 439.67
    random.seed(seed)
    np.random.seed(seed)

    # 运行禁忌搜索
    print("\n=== 运行禁忌搜索算法 ===")
    ts_start = datetime.now()
    ts_tsp = TabuSearchTSP(
        coords,
        tabu_size=250,
        max_iter=5000,
        neighbor_size=300,
        diversification_freq=200,
        aspiration_ratio=1.02,
        k_nearest=10,
    )
    ts_route, ts_dist, ts_history = ts_tsp.solve()
    ts_end = datetime.now()

    # 运行模拟退火
    print("\n=== 运行模拟退火算法 ===")
    sa_start = datetime.now()
    sa_tsp = SimulatedAnnealingTSP(
        coords, initial_temp=10000, cooling_rate=0.003, max_iter=5000, k_nearest=10
    )
    sa_route, sa_dist, sa_history = sa_tsp.solve()
    sa_end = datetime.now()

    # 验证解决方案
    print("\n=== 验证解决方案 ===")
    print(f"禁忌搜索解验证: {'通过' if verify_route(coords, ts_route) else '失败'}")
    print(f"模拟退火解验证: {'通过' if verify_route(coords, sa_route) else '失败'}")

    # 结果对比
    print("\n=== 算法性能对比 ===")
    print(f"禁忌搜索 - 最佳距离: {ts_dist:.2f}, 耗时: {ts_end - ts_start}")
    print(f"模拟退火 - 最佳距离: {sa_dist:.2f}, 耗时: {sa_end - sa_start}")
    print(f"与最优解(426)的差距:")
    print(f"  禁忌搜索: {(ts_dist - 426) / 426 * 100:.2f}%")
    print(f"  模拟退火: {(sa_dist - 426) / 426 * 100:.2f}%")

    # 保存可视化结果
    plot_route(
        coords,
        ts_route,
        f"Tabu Search (Distance: {ts_dist:.2f})",
        "output/ts_route.png",
    )
    plot_route(
        coords,
        sa_route,
        f"Simulated Annealing (Distance: {sa_dist:.2f})",
        "output/sa_route.png",
    )

    # 绘制收敛曲线对比
    plt.figure(figsize=(12, 6))
    plt.plot(ts_history, "b-", label=f"Tabu Search (Final: {ts_dist:.2f})")
    plt.plot(sa_history, "r-", label=f"Simulated Annealing (Final: {sa_dist:.2f})")
    plt.title("Algorithm Convergence Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/comparison.png")
    plt.close()

    print("\n结果已保存到output目录")


if __name__ == "__main__":
    main()
