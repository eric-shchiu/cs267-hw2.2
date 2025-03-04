#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Define MPI particle datatype
MPI_Datatype MPI_PARTICLE_TYPE;

// 2D decomposition parameters
static int grid_rows, grid_cols;    // Process grid dimensions
static int my_row, my_col;          // Current process coordinates
static double subdomain_xmin, subdomain_xmax; // Subdomain boundaries
static double subdomain_ymin, subdomain_ymax;
static double subdomain_width, subdomain_height;

// Particle storage
std::vector<particle_t> local_particles;  // Particles owned by this process
std::vector<particle_t> ghost_particles;  // Ghost particles from neighbors

// Neighbor process ranks (8 directions)
enum Neighbor { LEFT, RIGHT, TOP, BOTTOM, TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };
int neighbors[8] = { -1 };

// Modified apply_force with atomic operations
void apply_force(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff) return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;

    #pragma omp atomic
    particle.ax += coef * dx;
    #pragma omp atomic
    particle.ay += coef * dy;
}

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

// Custom MPI type creation for particle_t
void create_mpi_particle_type() {
    int block_lengths[6] = {1,1,1,1,1,1};
    MPI_Datatype types[6] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                            MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[6];
    
    offsets[0] = offsetof(particle_t, x);
    offsets[1] = offsetof(particle_t, y);
    offsets[2] = offsetof(particle_t, vx);
    offsets[3] = offsetof(particle_t, vy);
    offsets[4] = offsetof(particle_t, ax);
    offsets[5] = offsetof(particle_t, id);
    
    MPI_Type_create_struct(6, block_lengths, offsets, types, &MPI_PARTICLE_TYPE);
    MPI_Type_commit(&MPI_PARTICLE_TYPE);
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    create_mpi_particle_type();

    // Create 2D process grid (as close to square as possible)
    grid_rows = static_cast<int>(sqrt(num_procs));
    while(num_procs % grid_rows != 0) grid_rows--;
    grid_cols = num_procs / grid_rows;
    
    // Calculate process coordinates
    my_row = rank / grid_cols;
    my_col = rank % grid_cols;

    // Calculate subdomain boundaries
    subdomain_width = size / grid_cols;
    subdomain_height = size / grid_rows;
    subdomain_xmin = my_col * subdomain_width;
    subdomain_xmax = (my_col == grid_cols-1) ? size : subdomain_xmin + subdomain_width;
    subdomain_ymin = my_row * subdomain_height;
    subdomain_ymax = (my_row == grid_rows-1) ? size : subdomain_ymin + subdomain_height;

    // Initialize neighbor ranks
    neighbors[LEFT]         = (my_col > 0)       ? rank - 1 : -1;
    neighbors[RIGHT]        = (my_col < grid_cols-1) ? rank + 1 : -1;
    neighbors[TOP]          = (my_row > 0)       ? rank - grid_cols : -1;
    neighbors[BOTTOM]       = (my_row < grid_rows-1) ? rank + grid_cols : -1;
    neighbors[TOP_LEFT]     = (my_col > 0 && my_row > 0) ? rank - grid_cols - 1 : -1;
    neighbors[TOP_RIGHT]    = (my_col < grid_cols-1 && my_row > 0) ? rank - grid_cols + 1 : -1;
    neighbors[BOTTOM_LEFT]  = (my_col > 0 && my_row < grid_rows-1) ? rank + grid_cols - 1 : -1;
    neighbors[BOTTOM_RIGHT] = (my_col < grid_cols-1 && my_row < grid_rows-1) ? rank + grid_cols + 1 : -1;

    // Distribute initial particles
    for(int i = 0; i < num_parts; ++i) {
        if(parts[i].x >= subdomain_xmin && parts[i].x < subdomain_xmax &&
           parts[i].y >= subdomain_ymin && parts[i].y < subdomain_ymax) {
            local_particles.push_back(parts[i]);
        }
    }
}

void exchange_particles(int rank, int num_procs, double size) {
    const int tag = 0;
    MPI_Request send_requests[8], recv_requests[8];
    std::vector<particle_t> send_buffers[8];
    std::vector<particle_t> recv_buffers[8];

    // Classify particles to be sent to neighbors
    for(auto it = local_particles.begin(); it != local_particles.end();) {
        particle_t& p = *it;
        int dir = -1;

        if(p.x < subdomain_xmin) {
            if(p.y < subdomain_ymin)        dir = TOP_LEFT;
            else if(p.y >= subdomain_ymax)  dir = BOTTOM_LEFT;
            else                            dir = LEFT;
        }
        else if(p.x >= subdomain_xmax) {
            if(p.y < subdomain_ymin)        dir = TOP_RIGHT;
            else if(p.y >= subdomain_ymax)  dir = BOTTOM_RIGHT;
            else                            dir = RIGHT;
        }
        else if(p.y < subdomain_ymin)       dir = TOP;
        else if(p.y >= subdomain_ymax)      dir = BOTTOM;

        if(dir != -1 && neighbors[dir] != -1) {
            send_buffers[dir].push_back(p);
            it = local_particles.erase(it);
        } else {
            ++it;
        }
    }

    // Non-blocking sends
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;
        
        MPI_Isend(send_buffers[dir].data(), send_buffers[dir].size(),
                 MPI_PARTICLE_TYPE, neighbors[dir], tag, MPI_COMM_WORLD, &send_requests[dir]);
    }

    // Non-blocking receives
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;
        
        MPI_Status status;
        MPI_Probe(neighbors[dir], tag, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_PARTICLE_TYPE, &count);
        recv_buffers[dir].resize(count);
        
        MPI_Irecv(recv_buffers[dir].data(), count, MPI_PARTICLE_TYPE,
                 neighbors[dir], tag, MPI_COMM_WORLD, &recv_requests[dir]);
    }

    // Process received particles
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;

        MPI_Wait(&recv_requests[dir], MPI_STATUS_IGNORE);
        for(auto& p : recv_buffers[dir]) {
            // Classify as local or ghost particle
            if(p.x >= subdomain_xmin && p.x < subdomain_xmax &&
               p.y >= subdomain_ymin && p.y < subdomain_ymax) {
                local_particles.push_back(p);
            } else {
                ghost_particles.push_back(p);
            }
        }
    }

    // Ensure all sends complete
    MPI_Waitall(8, send_requests, MPI_STATUSES_IGNORE);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Combine local and ghost particles for force computation
    std::vector<particle_t> all_particles(local_particles);
    all_particles.insert(all_particles.end(), ghost_particles.begin(), ghost_particles.end());

    // Reset accelerations
    for(auto& p : all_particles) {
        p.ax = p.ay = 0.0;
    }

    // Bin particles for neighbor search (similar to serial version)
    double bin_size = cutoff;
    int num_bins_x = static_cast<int>((subdomain_xmax - subdomain_xmin)/bin_size) + 1;
    int num_bins_y = static_cast<int>((subdomain_ymax - subdomain_ymin)/bin_size) + 1;
    std::vector<std::vector<std::vector<int>>> bins(num_bins_x, std::vector<std::vector<int>>(num_bins_y));

    // Populate bins
    for(size_t i = 0; i < all_particles.size(); ++i) {
        int bin_x = static_cast<int>((all_particles[i].x - subdomain_xmin)/bin_size);
        int bin_y = static_cast<int>((all_particles[i].y - subdomain_ymin)/bin_size);
        bin_x = std::max(0, std::min(bin_x, num_bins_x-1));
        bin_y = std::max(0, std::min(bin_y, num_bins_y-1));
        bins[bin_x][bin_y].push_back(i);
    }

    // Compute forces (similar to serial version)
    for(int x = 0; x < num_bins_x; ++x) {
        for(int y = 0; y < num_bins_y; ++y) {
            auto& bin = bins[x][y];
            
            // Interactions within the bin
            for(size_t i = 0; i < bin.size(); ++i) {
                for(size_t j = i+1; j < bin.size(); ++j) {
                    particle_t& p1 = all_particles[bin[i]];
                    particle_t& p2 = all_particles[bin[j]];
                    apply_force(p1, p2);
                    apply_force(p2, p1);
                }
            }
            
            // Interactions with neighboring bins
            const int dx[] = {1, -1, 0, 0, 1, -1, 1, -1};
            const int dy[] = {0, 0, 1, -1, 1, 1, -1, -1};
            for(int d = 0; d < 8; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if(nx >= 0 && nx < num_bins_x && ny >= 0 && ny < num_bins_y) {
                    auto& nbin = bins[nx][ny];
                    for(size_t i : bin) {
                        for(size_t j : nbin) {
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
    exchange_particles(rank, num_procs, size);

    // Clear old ghost particles (will be refreshed in next step)
    ghost_particles.clear();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Gather all particles to rank 0
    int local_count = local_particles.size();
    std::vector<int> counts(num_procs), displs(num_procs);

    // Gather particle counts
    MPI_Gather(&local_count, 1, MPI_INT, 
              counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements
    if(rank == 0) {
        displs[0] = 0;
        for(int i = 1; i < num_procs; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    // Gather all particles
    MPI_Gatherv(local_particles.data(), local_count, MPI_PARTICLE_TYPE,
               parts, counts.data(), displs.data(), MPI_PARTICLE_TYPE,
               0, MPI_COMM_WORLD);

    // Sort particles by ID on rank 0
    if(rank == 0) {
        std::sort(parts, parts + num_parts, 
            [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
}
