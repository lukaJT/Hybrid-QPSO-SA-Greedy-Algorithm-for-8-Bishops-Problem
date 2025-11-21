import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# ===== 冲突检测  =====
def count_conflicts_fast(board):
    bishops = np.argwhere(board == 1)
    n = len(bishops)
    if n < 2:
        return 0, n
    
    # 统计每条对角线上的主教数量
    diag1_counts = {}  # 主对角线方向: r - c
    diag2_counts = {}  # 反对角线方向: r + c
    
    for r, c in bishops:
        d1 = r - c
        d2 = r + c
        diag1_counts[d1] = diag1_counts.get(d1, 0) + 1
        diag2_counts[d2] = diag2_counts.get(d2, 0) + 1
    
    # 计算冲突数：每条对角线上n个主教有n*(n-1)/2个冲突对
    conflicts = 0
    for count in diag1_counts.values():
        if count > 1:
            conflicts += count * (count - 1) // 2
    for count in diag2_counts.values():
        if count > 1:
            conflicts += count * (count - 1) // 2
    
    return conflicts, n

def score_board(board):
    conflicts, n = count_conflicts_fast(board)
    return n if conflicts == 0 else n - conflicts * 100

# ===== 贪心算法组件 =====
def greedy_optimize(board):
    current_board = board.copy()
    current_score = score_board(current_board)

    # 1. 尝试添加主教
    best_add_board = current_board
    best_add_score = current_score
    added = False
    for r, c in np.argwhere(current_board == 0):
        test_board = current_board.copy()
        test_board[r, c] = 1
        test_score = score_board(test_board)
        if test_score > best_add_score:
            best_add_score = test_score
            best_add_board = test_board
            added = True
    if added:
        return best_add_board

    # 2. 如果无法添加，尝试移除冲突最大的主教
    conflicts, _ = count_conflicts_fast(current_board)
    if conflicts == 0:
        return current_board

    best_remove_board = current_board
    best_remove_score = -np.inf
    for r, c in np.argwhere(current_board == 1):
        test_board = current_board.copy()
        test_board[r, c] = 0
        test_score = score_board(test_board)
        if test_score > best_remove_score:
            best_remove_score = test_score
            best_remove_board = test_board
    return best_remove_board

# ===== 模拟退火模块=====
class SimulatedAnnealing:
    def __init__(self, board_size=8, initial_temp=150, cooling_rate=0.95, min_temp=0.1):
        self.size = board_size
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def get_neighbor(self, board):
        neighbor = board.copy()
        # 策略：随机移动一个主教，或随机添加/删除一个主教
        bishops = np.argwhere(board == 1)
        empty_spots = np.argwhere(board == 0)

        if len(bishops) > 0 and len(empty_spots) > 0:
            if np.random.rand() < 0.6:
                # 移动一个主教
                r1, c1 = bishops[np.random.randint(0, len(bishops))]
                r2, c2 = empty_spots[np.random.randint(0, len(empty_spots))]
                neighbor[r1, c1] = 0
                neighbor[r2, c2] = 1
            elif np.random.rand() < 0.5:
                # 添加一个主教
                r, c = empty_spots[np.random.randint(0, len(empty_spots))]
                neighbor[r, c] = 1
            else:
                # 删除一个主教
                r, c = bishops[np.random.randint(0, len(bishops))]
                neighbor[r, c] = 0
        elif len(bishops) > 0:
            # 删除一个主教
            r, c = bishops[np.random.randint(0, len(bishops))]
            neighbor[r, c] = 0
        elif len(empty_spots) > 0:
            # 添加一个主教
            r, c = empty_spots[np.random.randint(0, len(empty_spots))]
            neighbor[r, c] = 1
        return neighbor

    def anneal(self, board, current_temp):
        current = board.copy()
        current_score = score_board(current)
        best = current.copy()
        best_score = current_score

        temp = current_temp
        while temp > self.min_temp:
            neighbor = self.get_neighbor(current)
            neighbor_score = score_board(neighbor)

            if neighbor_score > current_score or np.random.rand() < np.exp((neighbor_score - current_score) / temp):
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
                    if best_score >= 14 and count_conflicts_fast(best)[0] == 0:
                        break # 找到最优解，提前退出
            temp *= self.cooling_rate
        return best, best_score

# ===== 量子PSO核心=====
def quantum_update(x, pbest, gbest, alpha=0.5):
    # 生成一个介于pbest和gbest之间的吸引子
    attractor = np.where(np.random.rand(*x.shape) < alpha, pbest, gbest)
    # 围绕吸引子进行随机扰动
    return np.where(np.random.rand(*x.shape) < 0.5, attractor, x)

# ===== 并行处理的粒子更新函数 =====
def update_particle(args):
    i, position, pbest, gbest, alpha, sa_optimizer, current_temp, t = args
    
    # QPSO更新
    new_position = quantum_update(position, pbest, gbest, alpha)

    # SA优化 (每两代进行一次)
    if t % 2 == 0:
        new_position, _ = sa_optimizer.anneal(new_position, current_temp)

    # 评估
    current_score = score_board(new_position)
    
    return i, new_position, current_score

# ===== 增强混合QPSO + SA + 贪心 =====
def solve_bishops_hybrid_sa(n_particles=100, max_iter=300,alpha=0.5,
                            greedy_interval=5, sa_iterations=5):
    print("\n" + "="*60)
    print(f"三层混合算法 (QPSO + SA + 贪心) 求解8-Bishops问题")
    print(f"粒子数:{n_particles} | 迭代:{max_iter} | α={alpha}")
    print("="*60)

    sa_optimizer = SimulatedAnnealing(initial_temp=200, cooling_rate=0.98)

    # 1. QPSO初始化 - 使用更高效的方式
    positions = np.random.randint(0, 2, (n_particles, 8, 8), dtype=np.int8)
    # 初始化时主教数量不要太多
    for i in range(n_particles):
        positions[i] = np.where(np.random.rand(8, 8) < 0.2, 1, 0).astype(np.int8)

    pbest = positions.copy()
    pbest_scores = np.array([score_board(p) for p in pbest])
    gbest_idx = pbest_scores.argmax()
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    optimal_set = set()
    history = []
    solution_discovery_points = []  # 记录解发现的轮数

    # 2. 主循环 - 添加tqdm进度条
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for t in tqdm(range(max_iter), desc="算法迭代进度", unit="代", ncols=100):
            # 计算当前SA温度
            current_sa_temp = sa_optimizer.initial_temp * (sa_optimizer.cooling_rate ** t)

            # 准备并行任务参数
            tasks = []
            for i in range(n_particles):
                task_args = (i, positions[i], pbest[i], gbest, alpha, sa_optimizer, current_sa_temp, t)
                tasks.append(task_args)

            # 并行处理所有粒子
            futures = {executor.submit(update_particle, task): i for i, task in enumerate(tasks)}
            
            # 收集结果并更新
            for future in as_completed(futures):
                i, new_position, current_score = future.result()
                
                # 更新position和评估
                positions[i] = new_position

                # 更新pbest和gbest
                if current_score > pbest_scores[i]:
                    pbest[i] = new_position.copy()
                    pbest_scores[i] = current_score
                    if current_score == 14 and count_conflicts_fast(new_position)[0] == 0:
                        # 添加当前解到最优解集合
                        solution_tuple = tuple(new_position.flatten())
                        if solution_tuple not in optimal_set:
                            optimal_set.add(solution_tuple)
                            solution_discovery_points.append(t+1)  # 记录发现轮数
                            print(f"  发现新的最优解 #{len(optimal_set)} 于迭代 {t+1}!")

                if current_score > gbest_score:
                    print(f"  新全局最优: 分数 {gbest_score:.1f} -> {current_score:.1f}, 迭代: {t+1}")
                    gbest = new_position.copy()
                    gbest_score = current_score

            # 贪心精炼 (对gbest)
            if t % greedy_interval == 0:
                optimized_gbest = greedy_optimize(gbest)
                optimized_score = score_board(optimized_gbest)
                if optimized_score > gbest_score:
                    print(f"  贪心优化成功: 分数从 {gbest_score:.1f} 提升到 {optimized_score:.1f}")
                    gbest = optimized_gbest
                    gbest_score = optimized_score
                    if gbest_score == 14 and count_conflicts_fast(gbest)[0] == 0:
                        solution_tuple = tuple(gbest.flatten())
                        if solution_tuple not in optimal_set:
                            optimal_set.add(solution_tuple)
                            solution_discovery_points.append(t+1)  # 记录发现轮数
                            print(f"  发现新的最优解 #{len(optimal_set)} 通过贪心优化!")

            # 记录历史
            history.append((t+1, gbest_score))

            # 提前终止条件 - 但继续寻找更多解
            if gbest_score >= 14 and count_conflicts_fast(gbest)[0] == 0:
                print(f" 在第 {t+1} 次迭代找到理论最优解！继续搜索更多解...")

    # 3. 最终后处理
    print("\n应用最终后处理...")
    final_gbest = greedy_optimize(gbest)
    final_score = score_board(final_gbest)
    if final_score > gbest_score:
        print(f"  最终贪心优化成功: 分数从 {gbest_score:.1f} 提升到 {final_score:.1f}")
        gbest = final_gbest
        gbest_score = final_score

    # 检查最终解是否为最优解
    if gbest_score >= 14 and count_conflicts_fast(gbest)[0] == 0:
        solution_tuple = tuple(gbest.flatten())
        if solution_tuple not in optimal_set:
            optimal_set.add(solution_tuple)
            solution_discovery_points.append(max_iter)  # 记录发现轮数
            print(f"  发现最终最优解 #{len(optimal_set)} 通过最终贪心优化!")

    # 4. 输出结果
    conflicts, n_bishops = count_conflicts_fast(gbest)
    print("\n" + "="*60)
    print(f"最终解: 主教数={n_bishops}, 冲突数={conflicts}, 评分={gbest_score:.1f}")
    print(f"理论最优: 14个主教且无冲突")
    print(f"达到最优: {'✓' if gbest_score >= 14 and conflicts == 0 else '✗'}")
    print(f"不同最优解数量: {len(optimal_set)}")
    print("="*60)
    print("最优棋盘布局:")
    print(gbest)

    return gbest, optimal_set, history, solution_discovery_points

# ===== 可视化函数 =====
def visualize_single_board(board, title, ax):
    # 绘制棋盘
    for i in range(8):
        for j in range(8):
            color = '#DDBB88' if (i+j)%2 else '#FFFFFF'
            ax.add_patch(mpatches.Rectangle((j, 7-i), 1, 1, color=color))
    
    # 绘制主教
    for r, c in np.argwhere(board == 1):
        ax.plot(c+0.5, 7-r+0.5, 'ks', markersize=15, markerfacecolor='black',
                markeredgewidth=2, markeredgecolor='red')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    conflicts, n = count_conflicts_fast(board)
    ax.set_title(f'{n} Bishops\n{conflicts} Conflicts', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.5, color='black')

def visualize_all_solutions_and_convergence(best_board, solutions, history, solution_discovery_points):
    solutions_list = list(solutions)
    
    # 计算子图布局
    n_solutions = len(solutions_list)
    n_cols = min(5, max(1, n_solutions))  # 最多5列
    n_rows = (n_solutions + n_cols - 1) // n_cols  # 计算需要的行数
    
    # 创建总图
    total_plots = n_solutions + 1  # 解的数量 + 收敛图
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows + 2))
    
    # 绘制所有解
    for idx, solution_tuple in enumerate(solutions_list):
        solution_board = np.array(solution_tuple).reshape(8, 8)
        ax = fig.add_subplot(n_rows + 1, n_cols, idx + 1)  # +1 for convergence plot
        visualize_single_board(solution_board, f'Solution {idx+1}', ax)
    
    # 绘制收敛曲线
    ax_conv = fig.add_subplot(n_rows + 1, 1, n_rows + 1)
    if history:
        iterations, scores = zip(*history)
        ax_conv.plot(iterations, scores, 'b-', linewidth=2, marker='o', markersize=3, label='Best Score')
        ax_conv.axhline(y=14, color='r', linestyle='--', linewidth=2, label='Theoretical Optimum (14)')
        
        # 标注最优解发现点
        if solution_discovery_points:
            discovery_scores = []
            for point in solution_discovery_points:
                # 找到对应轮数的分数
                for iter_num, score in history:
                    if iter_num == point:
                        discovery_scores.append(score)
                        break
            ax_conv.scatter(solution_discovery_points, discovery_scores, 
                           color='green', s=50, zorder=5, label='Optimal Solutions Found')
        
        ax_conv.fill_between(iterations, 0, scores, alpha=0.2, color='blue')
        ax_conv.set_xlabel('Iteration', fontsize=12)
        ax_conv.set_ylabel('Global Best Score', fontsize=12)
        ax_conv.set_title('Optimized Hybrid Algorithm Convergence Curve', fontsize=14, fontweight='bold')
        ax_conv.legend(loc='lower right')
        ax_conv.grid(True, alpha=0.3)
        ax_conv.set_xlim(1, len(iterations))

    plt.tight_layout()
    plt.savefig('8_bishops_all_solutions_and_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印所有找到的最优解
    print(f"\n找到的 {len(solutions)} 个不同最优解：")
    for idx, solution_tuple in enumerate(list(solutions)):
        solution_board = np.array(solution_tuple).reshape(8, 8)
        conflicts, n_bishops = count_conflicts_fast(solution_board)
        print(f"\n解 #{idx+1}: {n_bishops} 个主教，{conflicts} 个冲突")
        print(solution_board)

# ===== 运行与可视化 =====
if __name__ == "__main__":
    np.random.seed(42)

    best_board, solutions, history, solution_discovery_points = solve_bishops_hybrid_sa(
        n_particles=120,
        max_iter=300,
        alpha=0.6,
        greedy_interval=4
    )

    # 可视化所有解和收敛曲线
    visualize_all_solutions_and_convergence(best_board, solutions, history, solution_discovery_points)