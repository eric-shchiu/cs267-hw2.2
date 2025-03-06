#include "common.h"
#include <mpi.h>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <iostream>

static double bin_size = cutoff; // bin size equals cutoff
static int num_bins_x, num_bins_y;
static std::vector<std::vector<std::vector<int>>> bins;

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

// Determine optimal grid decomposition
void determine_grid_dimensions(int num_procs, int &grid_rows, int &grid_cols) {
  
    // Iterate from sqrt_p down to 1 to find the best grid configuration,
    // ensuring that the number of processes in the grid is maximized
    // and the difference between the number of rows and columns donot exceed sqrt_p.
    int sqrt_p = static_cast<int>(sqrt(num_procs));
    int best_rows = 1, best_cols = num_procs;
    int best_num = 1;

    for (int rows = sqrt_p; rows >= 1; --rows) {
        int cols = num_procs / rows;
        if (rows * cols > num_procs) continue;

        // Relaxed condition: only check if rows * cols is better
        if (rows * cols > best_num) {
            best_num = rows * cols;
            best_rows = rows;
            best_cols = cols;
        }
    }
    // Handle prime case explicitly
    if (best_num == 1){
        best_rows = 1;
        best_cols = num_procs;
    }

    grid_rows = best_rows;
    grid_cols = best_cols;
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
    // create_mpi_particle_type();
    // std::cout << "simulation initiated" << std::endl;

    // Create 2D process grid (row-major order)
    determine_grid_dimensions(num_procs, grid_rows, grid_cols);
    my_row = rank / grid_cols;
    my_col = rank % grid_cols;

    bin_size = cutoff;
    num_bins_x = static_cast<int>(size / bin_size) + 1;
    num_bins_y = static_cast<int>(size / bin_size) + 1;
    bins.resize(num_bins_x, std::vector<std::vector<int>>(num_bins_y));

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
}

// Particle exchange with neighbors
void exchange_particles() {
    constexpr int tag = 0;
    MPI_Request send_reqs[8], recv_reqs[8];
    std::vector<particle_t> send_buf[8];
    std::vector<particle_t> recv_buf[8];

        // Clear the bins
        for (auto& row : bins) {
            for (auto& bin : row) {
                bin.clear();
            }
        }
    
        // Assign the particles to bins
        for (int i = 0; i < num_parts; ++i) {
            int bin_x = static_cast<int>(parts[i].x / bin_size);
            int bin_y = static_cast<int>(parts[i].y / bin_size);
            bin_x = std::max(0, std::min(bin_x, num_bins_x - 1));
            bin_y = std::max(0, std::min(bin_y, num_bins_y - 1));
            bins[bin_x][bin_y].push_back(i);
        }
    
        // Reset acceleration
        for (int i = 0; i < num_parts; ++i) {
            parts[i].ax = 0.0;
            parts[i].ay = 0.0;
        }

        if(!is_combined(p)) {
            combined_particles.erase(it);
        } else {
            ++it;
        }
    }

    // Non-blocking sends
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;
        
        MPI_Isend(send_buf[dir].data(), send_buf[dir].size(),
                 PARTICLE, neighbors[dir], tag, MPI_COMM_WORLD, &send_reqs[dir]);
        // std::cout << "Sending...\n" << std::endl;
    }

    // Non-blocking receives
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;

        MPI_Status status;
        int count;
        MPI_Probe(neighbors[dir], tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, PARTICLE, &count);
        // std::cout << "Probe...\n" << std::endl;
        
        if(count > 0) {
            recv_buf[dir].resize(count);
            MPI_Irecv(recv_buf[dir].data(), count, PARTICLE,
                     neighbors[dir], tag, MPI_COMM_WORLD, &recv_reqs[dir]);
            // std::cout << "Recieving...\n" << std::endl;
        }
    }

    // Finalize sends
    for(int dir = 0; dir < 8; ++dir) {
            if(neighbors[dir] != -1 && !send_buf[dir].empty()) {
                MPI_Wait(&send_reqs[dir], MPI_STATUS_IGNORE);
            }
        }

    // Process received data
    for(int dir = 0; dir < 8; ++dir) {
        if(neighbors[dir] == -1) continue;

        if(recv_buf[dir].size() > 0) {
            // std::cout << "Waiting...\n" << std::endl;
            MPI_Wait(&recv_reqs[dir], MPI_STATUS_IGNORE);
            combined_particles.insert(combined_particles.end(),
                                     recv_buf[dir].begin(),
                                     recv_buf[dir].end());
            // std::cout << "Continueing...\n" << std::endl;
        }
    }
}

// Main simulation step
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Combine particles for computation
    combined_particles = local_particles;
    combined_particles.insert(combined_particles.end(),
                             ghost_particles.begin(),
                             ghost_particles.end());

    // Reset accelerations
    for(auto& p : local_particles) {
        p.ax = p.ay = 0.0;
    }

    // Bin particles for neighbor search
    const double bin_size = cutoff;
    const int bins_x = static_cast<int>((sub_xmax - sub_xmin)/bin_size) + 1;
    const int bins_y = static_cast<int>((sub_ymax - sub_ymin)/bin_size) + 1;
    std::vector<std::vector<std::vector<int>>> bins(bins_x, std::vector<std::vector<int>>(bins_y));

    // Populate bins
    for(size_t i=0; i<combined_particles.size(); ++i) {
        const double rel_x = combined_particles[i].x - sub_xmin;
        const double rel_y = combined_particles[i].y - sub_ymin;
        int x = std::max(0, std::min(static_cast<int>(rel_x/bin_size), bins_x-1));
        int y = std::max(0, std::min(static_cast<int>(rel_y/bin_size), bins_y-1));
        bins[x][y].push_back(i);
    }

    // Compute forces
    for(int x=0; x<bins_x; ++x) {
        for(int y=0; y<bins_y; ++y) {
            auto& bin = bins[x][y];
            
            // Intra-bin interactions
            for(size_t i=0; i<bin.size(); ++i) {
                for(size_t j=i+1; j<bin.size(); ++j) {
                    particle_t& p1 = combined_particles[bin[i]];
                    particle_t& p2 = combined_particles[bin[j]];
                    apply_force(p1, p2);
                    apply_force(p2, p1);
                }
    
                // Interactions with right bin
                if (x + 1 < num_bins_x) {
                    auto& right_bin = bins[x + 1][y];
                    for (int pi : current_bin) {
                        for (int pj : right_bin) {
                            apply_force(parts[pi], parts[pj]);
                            apply_force(parts[pj], parts[pi]);
                        }
                    }
                }
    
                // Interactions with bottom bin
                if (y + 1 < num_bins_y) {
                    auto& bottom_bin = bins[x][y + 1];
                    for (int pi : current_bin) {
                        for (int pj : bottom_bin) {
                            apply_force(parts[pi], parts[pj]);
                            apply_force(parts[pj], parts[pi]);
                        }
                    }
                }
    
                // Interactions with bottom-right bin
                if (x + 1 < num_bins_x && y + 1 < num_bins_y) {
                    auto& br_bin = bins[x + 1][y + 1];
                    for (int pi : current_bin) {
                        for (int pj : br_bin) {
                            apply_force(parts[pi], parts[pj]);
                            apply_force(parts[pj], parts[pi]);
                        }
                    }
                }
                
                // Interactions with bottom-left bin
                if (x - 1 >= 0 && y + 1 < num_bins_y) {
                    auto& bl_bin = bins[x - 1][y + 1];
                    for (int pi : current_bin) {
                        for (int pj : bl_bin) {
                            apply_force(parts[pi], parts[pj]);
                            apply_force(parts[pj], parts[pi]);
                        }
                    }
                }
            }
        }
    }

    // Move only local particles
    for(auto& p : local_particles) {
        move(p, size);
    }

    // Perform particle migration
    exchange_particles();
    classify_particles();
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
