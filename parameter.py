import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from main import generate_initial_solution, read_tsp_file


class TabuSearchTSP:
    """禁忌搜索算法解决TSP问题"""

    def __init__(
        self,
        coords,
        initial_solution,
        tabu_size=50,  # 禁忌表大小
        max_iter=2000,  # 最大迭代次数
        neighbor_size=100,  # 邻域大小
        diversification_freq=100,  # 多样化频率
        aspiration_ratio=1.05,  # 渴望准则比率
        k_nearest=10,  # 考虑的近邻数量
    ):
        """初始化禁忌搜索参数"""
        self.coords = coords
        self.initial_solution = initial_solution
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
        # 初始解
        current_route = self.initial_solution
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
                # print(f"在迭代 {iteration} 进行多样化")
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
                    # print(f"迭代 {iteration}: 新的最佳距离: {best_dist:.2f}")

            # 记录历史
            history.append(best_dist)

            # 进度输出
            # if iteration % 100 == 0:
            # print(f"迭代 {iteration}, 当前最佳: {best_dist:.2f}")

        # 最终局部搜索
        # print("运行最终的2-opt局部搜索...")
        improved = True
        while improved:
            improved = False
            neighbors = self.get_neighbors(best_route)
            for neighbor, _, dist in neighbors:
                if dist < best_dist:
                    best_route = neighbor
                    best_dist = dist
                    improved = True
                    # print(f"最终改进: {best_dist:.2f}")
                    break

        return best_route, best_dist, history


# 生成随机RGB颜色
def random_color():
    return (random.random(), random.random(), random.random())


def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    # 读取TSP数据文件
    filename = "eil51.tsp"
    coords = read_tsp_file(filename)

    # 设置随机种子以保证可重复性
    seeds = [int(s) for s in np.random.randint(10, 1000000, size=5)]
    random.seed(seeds[0])
    np.random.seed(seeds[0])

    # 生成初始解
    initial_solution = generate_initial_solution(coords)

    k_nearest_dist = {}

    for k_nearest in range(10, 51, 5):
        print(f"运行禁忌搜索(k_nearest={k_nearest})...")
        k_nearest_dist[k_nearest] = 0
        for seed in seeds:
            # 重置种子
            random.seed(seed)
            np.random.seed(seed)

            # 运行禁忌搜索
            ts_start = datetime.now()
            ts_tsp = TabuSearchTSP(
                coords,
                initial_solution,
                tabu_size=250,
                max_iter=5000,
                neighbor_size=100,
                diversification_freq=200,
                aspiration_ratio=1.02,
                k_nearest=k_nearest,
            )
            ts_route, ts_dist, ts_history = ts_tsp.solve()
            ts_end = datetime.now()
            print(
                f"禁忌搜索(k_nearest={k_nearest}) - 最佳距离: {ts_dist:.2f}, 耗时: {ts_end - ts_start}"
            )
            k_nearest_dist[k_nearest] += ts_dist
        k_nearest_dist[k_nearest] /= len(seeds)
        print(
            f"禁忌搜索(k_nearest={k_nearest}) - 平均最佳距离: {k_nearest_dist[k_nearest]:.2f}"
        )

    # 绘制收敛曲线对比
    # 邻近数量对比
    plt.figure(figsize=(12, 6))
    plt.plot(k_nearest_dist.keys(), k_nearest_dist.values(), "b-")
    plt.title("Comparison")
    plt.xlabel("k_nearist")
    plt.ylabel("Distance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/parameter.png")
    plt.close()

    print("\n结果已保存到output目录")


if __name__ == "__main__":
    main()
