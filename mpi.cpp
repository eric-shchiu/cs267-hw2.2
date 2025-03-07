#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

// 2D decomposition parameters
static int grid_rows, grid_cols;
static int my_row, my_col;
static double sub_xmin, sub_xmax, sub_ymin, sub_ymax;
static double right_outer_margin, left_outer_margin, up_outer_margin, down_outer_margin;
static double right_inner_margin, left_inner_margin, up_inner_margin, down_inner_margin;

// Particle storage
std::vector<particle_t> local_particles;    // Particles owned by this process
std::vector<particle_t> ghost_particles;    // Ghost particles from neighbors
std::vector<particle_t> combined_particles; // Temporary storage during communication

// Neighbor process mapping
enum NeighborDir { 
    RIGHT = 0, TOP_RIGHT = 1, TOP = 2, TOP_LEFT = 3,
    LEFT = 4, BOTTOM_LEFT = 5, BOTTOM = 6, BOTTOM_RIGHT = 7 
};
int neighbors[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };

// Helper functions for particle classification
bool is_local(const particle_t& p) {
    return (p.x >= sub_xmin) && (p.x < sub_xmax) &&
           (p.y >= sub_ymin) && (p.y < sub_ymax);
}

bool is_combined(const particle_t& p) {
    return (p.x >= left_outer_margin) && (p.x < right_outer_margin) &&
           (p.y >= up_outer_margin) && (p.y < down_outer_margin);
}

bool in_ghost_zone(const particle_t& p) {
    return is_combined(p) && !is_local(p);
}

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void determine_grid_dimensions(int num_procs, int &grid_rows, int &grid_cols) {
    grid_rows = static_cast<int>(sqrt(num_procs));
    
    // Find the biggest row number that satisfies grid_rows ≤ sqrt(num_procs) and num_procs % grid_rows == 0
    while (grid_rows > 0) {
        if (num_procs % grid_rows == 0) {
            grid_cols = num_procs / grid_rows;
            break;
        }
        grid_rows--;
    }

    // If the number of processes is a prime number, use 1xnum_procs
    if (grid_rows == 0) {
        grid_rows = 1;
        grid_cols = num_procs;
    }
}

// Classify particles after communication
void classify_particles() {
    std::vector<particle_t> new_local, new_ghost;
    
    for(auto& p : combined_particles) {
        if(is_local(p)) {
            new_local.push_back(p);
        } else if(in_ghost_zone(p)) {
            new_ghost.push_back(p);
        }
        // Particles outside both zones are discarded
    }

    local_particles = std::move(new_local);
    ghost_particles = std::move(new_ghost);
}

// Initialize simulation domain and neighbors
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Create 2D process grid (row-major order)
    determine_grid_dimensions(num_procs, grid_rows, grid_cols);
    my_row = rank / grid_cols;
    my_col = rank % grid_cols;

    // Calculate subdomain boundaries
    const double sub_w = size / grid_cols;
    const double sub_h = size / grid_rows;
    sub_xmin = my_col * sub_w;
    sub_xmax = (my_col == grid_cols-1) ? size : sub_xmin + sub_w;
    sub_ymin = my_row * sub_h;
    sub_ymax = (my_row == grid_rows-1) ? size : sub_ymin + sub_h;

    // Calculate communication margins
    right_outer_margin = sub_xmax + cutoff;
    left_outer_margin = sub_xmin - cutoff;
    down_outer_margin = sub_ymax + cutoff;
    up_outer_margin = sub_ymin - cutoff;

    right_inner_margin = sub_xmax - cutoff;
    left_inner_margin = sub_xmin + cutoff;
    up_inner_margin = sub_ymin + cutoff;
    down_inner_margin = sub_ymax - cutoff;

    // Initialize neighbor processes
    neighbors[LEFT]         = (my_col > 0)       ? rank - 1 : -1;
    neighbors[RIGHT]        = (my_col < grid_cols-1) ? rank + 1 : -1;
    neighbors[TOP]          = (my_row > 0)       ? rank - grid_cols : -1;
    neighbors[BOTTOM]       = (my_row < grid_rows-1) ? rank + grid_cols : -1;
    neighbors[TOP_LEFT]     = (my_col > 0 && my_row > 0) ? rank - grid_cols - 1 : -1;
    neighbors[TOP_RIGHT]    = (my_col < grid_cols-1 && my_row > 0) ? rank - grid_cols + 1 : -1;
    neighbors[BOTTOM_LEFT]  = (my_col > 0 && my_row < grid_rows-1) ? rank + grid_cols - 1 : -1;
    neighbors[BOTTOM_RIGHT] = (my_col < grid_cols-1 && my_row < grid_rows-1) ? rank + grid_cols + 1 : -1;

    // Initial local particles and ghost particles
    local_particles.clear();
    ghost_particles.clear();
    for(int i = 0; i < num_parts; ++i) {
        if(is_local(parts[i])) {
            local_particles.push_back(parts[i]);
        }
        else if (in_ghost_zone(parts[i])) {
            ghost_particles.push_back(parts[i]);
        }
    }

    // Clear combined particles
    combined_particles.clear();
}

// Particle exchange with neighbors
void exchange_particles(int rank) {
    constexpr int tag = 0;
    MPI_Request send_reqs[8], recv_reqs[8];
    std::vector<particle_t> send_buf[8];
    std::vector<particle_t> recv_buf[8];

    // Prepare send buffers
    combined_particles = local_particles;
    for(auto it = combined_particles.begin(); it != combined_particles.end();) {
        particle_t& p = *it;

        const bool left_outer = (p.x > left_outer_margin);
        const bool right_outer = (p.x <= right_outer_margin);
        const bool up_outer = (p.y > up_outer_margin);
        const bool down_outer = (p.y <= down_outer_margin);

        const bool left_inner = (p.x < left_inner_margin);
        const bool right_inner = (p.x >= right_inner_margin);
        const bool up_inner = (p.y < up_inner_margin);
        const bool down_inner = (p.y >= down_inner_margin);

        if (right_inner && up_outer && down_outer && (neighbors[RIGHT] != -1)) {
            send_buf[RIGHT].push_back(p);
        }
        if (up_inner && right_inner && (neighbors[TOP_RIGHT] != -1)) {
            send_buf[TOP_RIGHT].push_back(p);
        }
        if (up_inner && left_outer && right_outer && (neighbors[TOP] != -1)) {
            send_buf[TOP].push_back(p);
        }
        if (up_inner && left_inner && (neighbors[TOP_LEFT] != -1)) {
            send_buf[TOP_LEFT].push_back(p);
        }
        if (left_inner && up_outer && down_outer && (neighbors[LEFT] != -1)) {
            send_buf[LEFT].push_back(p);
        }
        if (down_inner && left_inner && (neighbors[BOTTOM_LEFT] != -1)) {
            send_buf[BOTTOM_LEFT].push_back(p);
        }
        if (down_inner && left_outer && right_outer && (neighbors[BOTTOM] != -1)) {
            send_buf[BOTTOM].push_back(p);
        }
        if (down_inner && right_inner && (neighbors[BOTTOM_RIGHT] != -1)) {
            send_buf[BOTTOM_RIGHT].push_back(p);
        }

        if(!is_combined(p)) {
            it = combined_particles.erase(it);
        } else {
            ++it;
        }
    }

    // Non-blocking sends
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;
        
        MPI_Isend(send_buf[dir].data(), send_buf[dir].size(),
                 PARTICLE, neighbors[dir], tag,
                 MPI_COMM_WORLD, &send_reqs[dir]);
    }

    // Non-blocking receives
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;

        MPI_Status status;
        int count;
        MPI_Probe(neighbors[dir], tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &count);
        
        if(count > 0) {
            recv_buf[dir].resize(count);
            MPI_Irecv(recv_buf[dir].data(), count, PARTICLE,
                     neighbors[dir], tag, MPI_COMM_WORLD, &recv_reqs[dir]);
        }
    }

    // Process received data
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;

        if(recv_buf[dir].size() > 0) {
            MPI_Wait(&recv_reqs[dir], MPI_STATUS_IGNORE);
            combined_particles.insert(combined_particles.end(),
                                     recv_buf[dir].begin(),
                                     recv_buf[dir].end());
        }
    }

    // Finalize sends
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] != -1 && !send_buf[dir].empty()) {
            MPI_Wait(&send_reqs[dir], MPI_STATUS_IGNORE);
        }
    }
}

// Main simulation step
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Combined local and ghost particles for computation
    combined_particles = local_particles;
    combined_particles.insert(combined_particles.end(),
                             ghost_particles.begin(),
                             ghost_particles.end());
    const size_t local_count = local_particles.size();

    // Reset acceleration for local particles
    for (auto& p : local_particles) {
        p.ax = p.ay = 0.0;
    }

    // Setup binning parameters
    const double bin_size = cutoff;
    // const int bins_x = static_cast<int>((sub_xmax - sub_xmin) / bin_size) + 1;
    // const int bins_y = static_cast<int>((sub_ymax - sub_ymin) / bin_size) + 1;
    const int bins_x = static_cast<int>((right_outer_margin - left_outer_margin) / bin_size) + 1;
    const int bins_y = static_cast<int>((down_outer_margin - up_outer_margin) / bin_size) + 1;
    std::vector<std::vector<std::vector<int>>> bins(bins_x, std::vector<std::vector<int>>(bins_y));

    // asign particles to bins
    for (size_t i = 0; i < combined_particles.size(); ++i) {
        // const double rel_x = combined_particles[i].x - sub_xmin;
        // const double rel_y = combined_particles[i].y - sub_ymin;
        const double rel_x = combined_particles[i].x - left_outer_margin;
        const double rel_y = combined_particles[i].y - up_outer_margin;
        int x_bin = std::max(0, std::min(static_cast<int>(rel_x / bin_size), bins_x - 1));
        int y_bin = std::max(0, std::min(static_cast<int>(rel_y / bin_size), bins_y - 1));
        bins[x_bin][y_bin].push_back(i);
    }

    // Calculate forces
    for (int x = 0; x < bins_x; ++x) {
        for (int y = 0; y < bins_y; ++y) {
            auto& current_bin = bins[x][y];

            // Go through all the particles in the bin
            for (size_t i = 0; i < current_bin.size(); ++i) {
                const int idx_i = current_bin[i];
                particle_t& p1 = combined_particles[idx_i];
                const bool p1_is_local = idx_i < local_count;

                // Calculate forces between particles in the same bin
                for (size_t j = i + 1; j < current_bin.size(); ++j) {
                    const int idx_j = current_bin[j];
                    particle_t& p2 = combined_particles[idx_j];
                    const bool p2_is_local = idx_j < local_count;

                    // Apply force only when the force-receiving particle is local
                    if (p1_is_local) apply_force(p1, p2);
                    if (p2_is_local) apply_force(p2, p1);
                }
                
                // Skip if the particle is not local, do not receive force from particles in neighboring bins
                if (!p1_is_local) continue;

                // reveiving forces from particles in 8 neighboring bins
                const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
                const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
                for (int d = 0; d < 8; ++d) {
                    const int nx = x + dx[d];
                    const int ny = y + dy[d];
                    if (nx >= 0 && nx < bins_x && ny >= 0 && ny < bins_y) {
                        for (const int idx_j : bins[nx][ny]) {
                            particle_t& p2 = combined_particles[idx_j];
                            const bool p2_is_local = idx_j < local_count;

                            // receive force only when the force-receiving particle is local
                            if (p1_is_local) apply_force(p1, p2);
                        }
                    }
                }
            }
        }
    }

    // Only move local particles
    for (auto& p : local_particles) {
        move(p, size);
    }

    // Perform particle migration
    exchange_particles(rank);
    classify_particles();
}

// Data gathering for output
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int local_count = local_particles.size();
    std::vector<int> counts(num_procs), displs(num_procs);

    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        displs[0] = 0;
        for(int i=1; i<num_procs; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    MPI_Gatherv(local_particles.data(), local_count, PARTICLE,
               parts, counts.data(), displs.data(), PARTICLE,
               0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::sort(parts, parts + num_parts, 
            [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
}
