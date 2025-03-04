#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

// Define MPI particle type
MPI_Datatype MPI_PARTICLE_TYPE;

// 2D decomposition parameters
static int grid_rows, grid_cols;
static int my_row, my_col;
static double sub_xmin, sub_xmax, sub_ymin, sub_ymax;

// Particle storage
std::vector<particle_t> local_particles;
std::vector<particle_t> ghost_particles;

// Neighbor process mapping
enum Neighbor { LEFT, RIGHT, TOP, BOTTOM, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };
int neighbors[8] = { -1 };

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

// Create MPI particle type with proper offset handling
void create_mpi_particle_type() {
    static_assert(std::is_pod<particle_t>::value, "particle_t must be a POD type");

    const int num_fields = 6;
    MPI_Datatype types[num_fields] = { 
        MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
        MPI_DOUBLE, MPI_DOUBLE, MPI_INT 
    };
    int block_lengths[num_fields] = {1,1,1,1,1,1};
    MPI_Aint offsets[num_fields];

    offsets[0] = offsetof(particle_t, x);
    offsets[1] = offsetof(particle_t, y);
    offsets[2] = offsetof(particle_t, vx);
    offsets[3] = offsetof(particle_t, vy);
    offsets[4] = offsetof(particle_t, ax);
    offsets[5] = offsetof(particle_t, id);

    MPI_Type_create_struct(num_fields, block_lengths, offsets, types, &MPI_PARTICLE_TYPE);
    MPI_Type_commit(&MPI_PARTICLE_TYPE);
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    create_mpi_particle_type();

    // Create 2D process grid
    grid_rows = static_cast<int>(sqrt(num_procs));
    while(num_procs % grid_rows != 0) grid_rows--;
    grid_cols = num_procs / grid_rows;

    // Calculate process coordinates
    my_row = rank / grid_cols;
    my_col = rank % grid_cols;

    // Calculate subdomain boundaries
    const double sub_w = size / grid_cols;
    const double sub_h = size / grid_rows;
    sub_xmin = my_col * sub_w;
    sub_xmax = (my_col == grid_cols-1) ? size : sub_xmin + sub_w;
    sub_ymin = my_row * sub_h;
    sub_ymax = (my_row == grid_rows-1) ? size : sub_ymin + sub_h;

    // Initialize neighbor processes
    neighbors[LEFT]        = (my_col > 0)       ? rank - 1 : -1;
    neighbors[RIGHT]       = (my_col < grid_cols-1) ? rank + 1 : -1;
    neighbors[TOP]         = (my_row > 0)       ? rank - grid_cols : -1;
    neighbors[BOTTOM]      = (my_row < grid_rows-1) ? rank + grid_cols : -1;
    neighbors[TOP_LEFT]    = (my_col > 0 && my_row > 0) ? rank - grid_cols - 1 : -1;
    neighbors[TOP_RIGHT]   = (my_col < grid_cols-1 && my_row > 0) ? rank - grid_cols + 1 : -1;
    neighbors[BOTTOM_LEFT] = (my_col > 0 && my_row < grid_rows-1) ? rank + grid_cols - 1 : -1;
    neighbors[BOTTOM_RIGHT]= (my_col < grid_cols-1 && my_row < grid_rows-1) ? rank + grid_cols + 1 : -1;

    // Distribute initial particles
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

    // Prepare send buffers
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

    // Initiate non-blocking communication
    for(int dir=0; dir<8; ++dir) {
        if(neighbors[dir] == -1) continue;

        MPI_Isend(send_buf[dir].data(), send_buf[dir].size(),
                 MPI_PARTICLE_TYPE, neighbors[dir], tag, MPI_COMM_WORLD, &send_reqs[dir]);

        MPI_Status probe_status;
        MPI_Probe(neighbors[dir], tag, MPI_COMM_WORLD, &probe_status);
        int recv_count;
        MPI_Get_count(&probe_status, MPI_PARTICLE_TYPE, &recv_count);
        recv_buf[dir].resize(recv_count);
        
        MPI_Irecv(recv_buf[dir].data(), recv_count, MPI_PARTICLE_TYPE,
                 neighbors[dir], tag, MPI_COMM_WORLD, &recv_reqs[dir]);
    }

    // Process received particles
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

    // Ensure all sends complete
    MPI_Waitall(8, send_reqs, MPI_STATUSES_IGNORE);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Combine local and ghost particles
    std::vector<particle_t> all_particles = local_particles;
    all_particles.insert(all_particles.end(), ghost_particles.begin(), ghost_particles.end());

    // Reset accelerations
    for(auto& p : all_particles) {
        p.ax = p.ay = 0.0;
    }

    // Bin particles for neighbor search
    const double bin_size = cutoff;
    const int bins_x = static_cast<int>((sub_xmax - sub_xmin)/bin_size) + 1;
    const int bins_y = static_cast<int>((sub_ymax - sub_ymin)/bin_size) + 1;
    std::vector<std::vector<std::vector<int>>> bins(bins_x, std::vector<std::vector<int>>(bins_y));

    // Populate bins using max/min instead of clamp
    for(size_t i=0; i<all_particles.size(); ++i) {
        const double rel_x = all_particles[i].x - sub_xmin;
        const double rel_y = all_particles[i].y - sub_ymin;
        int x = static_cast<int>(rel_x / bin_size);
        int y = static_cast<int>(rel_y / bin_size);
        x = std::max(0, std::min(x, bins_x-1));  // Replaced clamp with max/min
        y = std::max(0, std::min(y, bins_y-1));  // Replaced clamp with max/min
        bins[x][y].push_back(i);
    }

    // Compute forces
    for(int x=0; x<bins_x; ++x) {
        for(int y=0; y<bins_y; ++y) {
            auto& bin = bins[x][y];
            
            // Intra-bin interactions
            for(size_t i=0; i<bin.size(); ++i) {
                for(size_t j=i+1; j<bin.size(); ++j) {
                    particle_t& p1 = all_particles[bin[i]];
                    particle_t& p2 = all_particles[bin[j]];
                    apply_force(p1, p2);
                    apply_force(p2, p1);
                }
            }
            
            // Inter-bin interactions
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

    // Move local particles
    for(auto& p : local_particles) {
        move(p, size);
    }

    // Exchange particles with neighbors
    exchange_particles(rank);

    // Clear ghost particles for next step
    ghost_particles.clear();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    int local_count = local_particles.size();
    std::vector<int> counts(num_procs), displs(num_procs);

    // Gather particle counts
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements
    if(rank == 0) {
        displs[0] = 0;
        for(int i=1; i<num_procs; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    // Gather all particles
    MPI_Gatherv(local_particles.data(), local_count, MPI_PARTICLE_TYPE,
               parts, counts.data(), displs.data(), MPI_PARTICLE_TYPE,
               0, MPI_COMM_WORLD);

    // Sort particles by ID on root
    if(rank == 0) {
        std::sort(parts, parts + num_parts, 
            [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
}
