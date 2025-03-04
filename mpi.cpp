#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>      // 添加offsetof所需头文件
#include <type_traits>  // 添加类型校验支持

// 定义MPI粒子类型
MPI_Datatype MPI_PARTICLE_TYPE;

// 2D分区参数
static int grid_rows, grid_cols;
static int my_row, my_col;
static double sub_xmin, sub_xmax, sub_ymin, sub_ymax;

// 粒子存储
std::vector<particle_t> local_particles;
std::vector<particle_t> ghost_particles;

// 邻居进程映射
enum Neighbor { LEFT, RIGHT, TOP, BOTTOM, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };
int neighbors[8] = { -1 };

// 创建MPI粒子类型（修正版）
void create_mpi_particle_type() {
    // 验证particle_t是否为POD类型
    static_assert(std::is_pod<particle_t>::value, "particle_t must be a POD type");

    // 定义类型结构
    const int num_fields = 6;
    MPI_Datatype types[num_fields] = { 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_INT 
    };
    int block_lengths[num_fields] = {1,1,1,1,1,1};
    MPI_Aint offsets[num_fields];

    // 获取字段偏移量（使用标准offsetof）
    offsets[0] = offsetof(particle_t, x);
    offsets[1] = offsetof(particle_t, y);
    offsets[2] = offsetof(particle_t, vx);
    offsets[3] = offsetof(particle_t, vy);
    offsets[4] = offsetof(particle_t, ax);
    offsets[5] = offsetof(particle_t, id);

    // 创建MPI数据类型
    MPI_Type_create_struct(num_fields, block_lengths, offsets, types, &MPI_PARTICLE_TYPE);
    MPI_Type_commit(&MPI_PARTICLE_TYPE);
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    create_mpi_particle_type();

    // 创建2D进程网格
    grid_rows = static_cast<int>(sqrt(num_procs));
    while(num_procs % grid_rows != 0) grid_rows--;
    grid_cols = num_procs / grid_rows;

    // 计算进程坐标
    my_row = rank / grid_cols;
    my_col = rank % grid_cols;

    // 计算子域边界
    const double sub_w = size / grid_cols;
    const double sub_h = size / grid_rows;
    sub_xmin = my_col * sub_w;
    sub_xmax = (my_col == grid_cols-1) ? size : sub_xmin + sub_w;
    sub_ymin = my_row * sub_h;
    sub_ymax = (my_row == grid_rows-1) ? size : sub_ymin + sub_h;

    // 初始化邻居进程
    neighbors[LEFT]        = (my_col > 0)       ? rank - 1 : -1;
    neighbors[RIGHT]       = (my_col < grid_cols-1) ? rank + 1 : -1;
    neighbors[TOP]         = (my_row > 0)       ? rank - grid_cols : -1;
    neighbors[BOTTOM]      = (my_row < grid_rows-1) ? rank + grid_cols : -1;
    neighbors[TOP_LEFT]    = (my_col > 0 && my_row > 0) ? rank - grid_cols - 1 : -1;
    neighbors[TOP_RIGHT]   = (my_col < grid_cols-1 && my_row > 0) ? rank - grid_cols + 1 : -1;
    neighbors[BOTTOM_LEFT] = (my_col > 0 && my_row < grid_rows-1) ? rank + grid_cols - 1 : -1;
    neighbors[BOTTOM_RIGHT]= (my_col < grid_cols-1 && my_row < grid_rows-1) ? rank + grid_cols + 1 : -1;

    // 初始粒子分配
    for(int i=0; i<num_parts; ++i) {
        const auto& p = parts[i];
        if(p.x >= sub_xmin && p.x < sub_xmax &&
           p.y >= sub_ymin && p.y < sub_ymax) {
            local_particles.push_back(p);
        }
    }
}

void exchange_particles(int rank) {
    const int tag = 0;
    MPI_Request send_reqs[8], recv_reqs[8];
    std::vector<particle_t> send_buf[8], recv_buf[8];

    // 1. 准备发送数据
    for(auto it=local_particles.begin(); it!=local_particles.end();) {
        particle_t& p = *it;
        int dir = -1;

        if(p.x < sub_xmin) {
            if(p.y < sub_ymin)        dir = TOP_LEFT;
            else if(p.y >= sub_ymax)  dir = BOTTOM_LEFT;
            else                      dir = LEFT;
        }
        else if(p.x >= sub_xmax) {
            if(p.y < sub_ymin)        dir = TOP_RIGHT;
            else if(p.y >= sub_ymax)  dir = BOTTOM_RIGHT;
            else                      dir = RIGHT;
        }
        else if(p.y < sub_ymin)       dir = TOP;
        else if(p.y >= sub_ymax)      dir = BOTTOM;

        if(dir != -1 && neighbors[dir] != -1) {
            send_buf[dir].push_back(p);
            it = local_particles.erase(it);
        } else {
            ++it;
        }
    }

    // 2. 发起非阻塞通信
    for(int dir=0; dir<8; ++dir) {
        if(neighbors[dir] == -1) continue;

        // 发送数据
        MPI_Isend(send_buf[dir].data(), send_buf[dir].size(),
                 MPI_PARTICLE_TYPE, neighbors[dir], tag, MPI_COMM_WORLD, &send_reqs[dir]);

        // 准备接收缓冲区
        MPI_Status probe_status;
        MPI_Probe(neighbors[dir], tag, MPI_COMM_WORLD, &probe_status);
        int recv_count;
        MPI_Get_count(&probe_status, MPI_PARTICLE_TYPE, &recv_count);
        recv_buf[dir].resize(recv_count);
        
        MPI_Irecv(recv_buf[dir].data(), recv_count, MPI_PARTICLE_TYPE,
                 neighbors[dir], tag, MPI_COMM_WORLD, &recv_reqs[dir]);
    }

    // 3. 处理接收数据
    for(int dir=0; dir<8; ++dir) {
        if(neighbors[dir] == -1) continue;

        MPI_Wait(&recv_reqs[dir], MPI_STATUS_IGNORE);
        for(auto& p : recv_buf[dir]) {
            if(p.x >= sub_xmin && p.x < sub_xmax &&
               p.y >= sub_ymin && p.y < sub_ymax) {
                local_particles.push_back(p);
            } else {
                ghost_particles.push_back(p);
            }
        }
    }

    // 4. 确保发送完成
    MPI_Waitall(8, send_reqs, MPI_STATUSES_IGNORE);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // 合并粒子集合
    std::vector<particle_t> all_particles = local_particles;
    all_particles.insert(all_particles.end(), ghost_particles.begin(), ghost_particles.end());

    // 重置加速度
    for(auto& p : all_particles) {
        p.ax = p.ay = 0.0;
    }

    // 分箱计算（类似串行版本）
    const double bin_size = cutoff;
    const int bins_x = static_cast<int>((sub_xmax - sub_xmin)/bin_size) + 1;
    const int bins_y = static_cast<int>((sub_ymax - sub_ymin)/bin_size) + 1;
    std::vector<std::vector<std::vector<int>>> bins(bins_x, std::vector<std::vector<int>>(bins_y));

    // 填充箱子
    for(size_t i=0; i<all_particles.size(); ++i) {
        const double rel_x = all_particles[i].x - sub_xmin;
        const double rel_y = all_particles[i].y - sub_ymin;
        int x = static_cast<int>(rel_x / bin_size);
        int y = static_cast<int>(rel_y / bin_size);
        x = std::clamp(x, 0, bins_x-1);
        y = std::clamp(y, 0, bins_y-1);
        bins[x][y].push_back(i);
    }

    // 计算作用力
    for(int x=0; x<bins_x; ++x) {
        for(int y=0; y<bins_y; ++y) {
            auto& bin = bins[x][y];
            
            // 本箱内部相互作用
            for(size_t i=0; i<bin.size(); ++i) {
                for(size_t j=i+1; j<bin.size(); ++j) {
                    particle_t& p1 = all_particles[bin[i]];
                    particle_t& p2 = all_particles[bin[j]];
                    apply_force(p1, p2);
                    apply_force(p2, p1);
                }
            }
            
            // 相邻箱子相互作用
            const int offsets[][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}, 
                                     {1,1}, {-1,1}, {1,-1}, {-1,-1}};
            for(const auto& off : offsets) {
                const int nx = x + off[0];
                const int ny = y + off[1];
                if(nx >=0 && nx < bins_x && ny >=0 && ny < bins_y) {
                    for(auto i : bin) {
                        for(auto j : bins[nx][ny]) {
                            if(i == j) continue;
                            particle_t& p1 = all_particles[i];
                            particle_t& p2 = all_particles[j];
                            apply_force(p1, p2);
                            apply_force(p2, p1);
                        }
                    }
                }
            }
        }
    }

    // 移动本地粒子
    for(auto& p : local_particles) {
        move(p, size);
    }

    // 交换粒子
    exchange_particles(rank);

    // 清理幽灵粒子
    ghost_particles.clear();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int local_count = local_particles.size();
    std::vector<int> counts(num_procs), displs(num_procs);

    // 收集粒子数量
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 计算位移
    if(rank == 0) {
        displs[0] = 0;
        for(int i=1; i<num_procs; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    // 收集粒子数据
    MPI_Gatherv(local_particles.data(), local_count, MPI_PARTICLE_TYPE,
               parts, counts.data(), displs.data(), MPI_PARTICLE_TYPE,
               0, MPI_COMM_WORLD);

    // 主进程排序
    if(rank == 0) {
        std::sort(parts, parts + num_parts, 
            [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
}
