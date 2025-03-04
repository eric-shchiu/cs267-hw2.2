#include "common.h"
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cmath>

// 2D processor grid dimensions
static int proc_grid_x, proc_grid_y;        // Processor grid dimensions
static int local_grid_x, local_grid_y;      // Coordinates of local subdomain in global grid
static double local_min_x, local_max_x;     // Local subdomain boundaries with ghost regions
static double local_min_y, local_max_y;

// Particle storage
static std::vector<particle_t> local_particles;  // Particles owned by this process
static std::vector<particle_t> ghost_particles;  // Ghost particles from neighbors

// Neighbor communication ranks (8 directions)
enum NeighborDir { LEFT, RIGHT, TOP, BOTTOM, 
                   TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT };
static int neighbor_ranks[8];  // Ranks of neighboring processes (-1 if non-existent)

// Custom MPI datatype for particle_t
static MPI_Datatype MPI_PARTICLE_TYPE;

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

// Helper function to create MPI particle datatype
void create_particle_type() {
    int blocklengths[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[5];
    
    offsets[0] = offsetof(particle_t, x);
    offsets[1] = offsetof(particle_t, y);
    offsets[2] = offsetof(particle_t, vx);
    offsets[3] = offsetof(particle_t, vy);
    offsets[4] = offsetof(particle_t, id);
    
    MPI_Type_create_struct(5, blocklengths, offsets, types, &MPI_PARTICLE_TYPE);
    MPI_Type_commit(&MPI_PARTICLE_TYPE);
}

// Initialize 2D decomposition and neighbor ranks
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    create_particle_type();

    // Create 2D processor grid (as close to square as possible)
    proc_grid_x = static_cast<int>(sqrt(num_procs));
    while (num_procs % proc_grid_x != 0) proc_grid_x--;
    proc_grid_y = num_procs / proc_grid_x;
    
    // Calculate local grid coordinates
    local_grid_y = rank / proc_grid_x;
    local_grid_x = rank % proc_grid_x;

    // Calculate local domain boundaries (with ghost regions)
    const double sub_size_x = size / proc_grid_x;
    const double sub_size_y = size / proc_grid_y;
    local_min_x = local_grid_x * sub_size_x - cutoff;
    local_max_x = (local_grid_x + 1) * sub_size_x + cutoff;
    local_min_y = local_grid_y * sub_size_y - cutoff;
    local_max_y = (local_grid_y + 1) * sub_size_y + cutoff;

    // Initialize neighbor ranks
    neighbor_ranks[LEFT]          = (local_grid_x > 0)            ? rank - 1 : -1;
    neighbor_ranks[RIGHT]         = (local_grid_x < proc_grid_x-1) ? rank + 1 : -1;
    neighbor_ranks[TOP]           = (local_grid_y > 0)            ? rank - proc_grid_x : -1;
    neighbor_ranks[BOTTOM]        = (local_grid_y < proc_grid_y-1) ? rank + proc_grid_x : -1;
    neighbor_ranks[TOP_LEFT]      = (neighbor_ranks[TOP] != -1 && neighbor_ranks[LEFT] != -1) 
                                   ? neighbor_ranks[TOP] - 1 : -1;
    neighbor_ranks[TOP_RIGHT]     = (neighbor_ranks[TOP] != -1 && neighbor_ranks[RIGHT] != -1) 
                                   ? neighbor_ranks[TOP] + 1 : -1;
    neighbor_ranks[BOTTOM_LEFT]   = (neighbor_ranks[BOTTOM] != -1 && neighbor_ranks[LEFT] != -1) 
                                   ? neighbor_ranks[BOTTOM] - 1 : -1;
    neighbor_ranks[BOTTOM_RIGHT]  = (neighbor_ranks[BOTTOM] != -1 && neighbor_ranks[RIGHT] != -1) 
                                   ? neighbor_ranks[BOTTOM] + 1 : -1;

    // Distribute initial particles
    for (int i = 0; i < num_parts; ++i) {
        if (parts[i].x >= local_min_x && parts[i].x < local_max_x &&
            parts[i].y >= local_min_y && parts[i].y < local_max_y) {
            local_particles.push_back(parts[i]);
        }
    }
}

// Exchange particles with neighboring domains
void exchange_ghost_particles(int rank, int num_procs) {
    const int num_neighbors = 8;
    MPI_Request requests[num_neighbors*4];
    int req_count = 0;
    std::vector<particle_t> send_buffers[num_neighbors];
    std::vector<particle_t> recv_buffers[num_neighbors];
    int send_counts[num_neighbors] = {0};
    int recv_counts[num_neighbors] = {0};

    // Step 1: Identify particles to send to each neighbor
    for (auto& p : local_particles) {
        // Predict next position
        double new_x = p.x + p.vx * dt;
        double new_y = p.y + p.vy * dt;
        
        // Check which neighbors need this particle
        for (int dir = 0; dir < num_neighbors; ++dir) {
            if (neighbor_ranks[dir] == -1) continue;
            
            // Define neighbor's physical boundaries
            int nx = local_grid_x, ny = local_grid_y;
            switch(dir) {
                case LEFT:          nx--; break;
                case RIGHT:         nx++; break;
                case TOP:           ny--; break;
                case BOTTOM:        ny++; break;
                case TOP_LEFT:     nx--; ny--; break;
                case TOP_RIGHT:     nx++; ny--; break;
                case BOTTOM_LEFT:  nx--; ny++; break;
                case BOTTOM_RIGHT: nx++; ny++; break;
            }
            
            double n_min_x = nx * (local_max_x - 2*cutoff)/proc_grid_x - cutoff;
            double n_max_x = (nx+1) * (local_max_x - 2*cutoff)/proc_grid_x + cutoff;
            double n_min_y = ny * (local_max_y - 2*cutoff)/proc_grid_y - cutoff;
            double n_max_y = (ny+1) * (local_max_y - 2*cutoff)/proc_grid_y + cutoff;

            if (new_x >= n_min_x && new_x < n_max_x &&
                new_y >= n_min_y && new_y < n_max_y) {
                send_buffers[dir].push_back(p);
            }
        }
    }

    // Step 2: Exchange particle counts with neighbors
    for (int dir = 0; dir < num_neighbors; ++dir) {
        if (neighbor_ranks[dir] == -1) continue;
        
        MPI_Isend(&send_counts[dir], 1, MPI_INT, neighbor_ranks[dir], 0, 
                 MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(&recv_counts[dir], 1, MPI_INT, neighbor_ranks[dir], 0,
                 MPI_COMM_WORLD, &requests[req_count++]);
    }
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    req_count = 0;

    // Step 3: Exchange particle data
    for (int dir = 0; dir < num_neighbors; ++dir) {
        if (neighbor_ranks[dir] == -1) continue;
        
        if (send_counts[dir] > 0) {
            MPI_Isend(send_buffers[dir].data(), send_counts[dir], MPI_PARTICLE_TYPE,
                     neighbor_ranks[dir], 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
        if (recv_counts[dir] > 0) {
            recv_buffers[dir].resize(recv_counts[dir]);
            MPI_Irecv(recv_buffers[dir].data(), recv_counts[dir], MPI_PARTICLE_TYPE,
                     neighbor_ranks[dir], 1, MPI_COMM_WORLD, &requests[req_count++]);
        }
    }
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Step 4: Merge received ghost particles
    ghost_particles.clear();
    for (int dir = 0; dir < num_neighbors; ++dir) {
        ghost_particles.insert(ghost_particles.end(), 
                              recv_buffers[dir].begin(), recv_buffers[dir].end());
    }
}

// Main simulation step
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Phase 1: Move local particles
    for (auto& p : local_particles) {
        move(p, size);
    }

    // Phase 2: Exchange ghost particles
    exchange_ghost_particles(rank, num_procs);

    // Phase 3: Compute forces (combine local and ghost particles)
    std::vector<particle_t> all_particles(local_particles);
    all_particles.insert(all_particles.end(), ghost_particles.begin(), ghost_particles.end());

    #pragma omp parallel for  // Hybrid OpenMP parallelization
    for (size_t i = 0; i < local_particles.size(); ++i) {
        local_particles[i].ax = 0.0;
        local_particles[i].ay = 0.0;
        
        for (size_t j = 0; j < all_particles.size(); ++j) {
            if (local_particles[i].id == all_particles[j].id) continue;
            apply_force(local_particles[i], all_particles[j]);
        }
    }

    // Phase 4: Remove particles that left local domain
    auto new_end = std::remove_if(local_particles.begin(), local_particles.end(),
        [&](const particle_t& p) {
            return p.x < (local_min_x + cutoff) || p.x >= (local_max_x - cutoff) ||
                   p.y < (local_min_y + cutoff) || p.y >= (local_max_y - cutoff);
        });
    local_particles.erase(new_end, local_particles.end());
}

// Collect all particles to rank 0 for saving
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Gather particle counts
    int local_count = local_particles.size();
    std::vector<int> counts(num_procs), displs(num_procs);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare displacement array on root
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; ++i) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    // Gather all particles
    MPI_Gatherv(local_particles.data(), local_count, MPI_PARTICLE_TYPE,
               parts, counts.data(), displs.data(), MPI_PARTICLE_TYPE, 0, MPI_COMM_WORLD);

    // Sort particles by ID on root
    if (rank == 0) {
        std::sort(parts, parts + num_parts, [](const particle_t& a, const particle_t& b) {
            return a.id < b.id;
        });
    }
}

// Cleanup MPI datatype
void simulation_finalize() {
    MPI_Type_free(&MPI_PARTICLE_TYPE);
}
