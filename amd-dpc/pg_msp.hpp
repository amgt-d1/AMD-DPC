#include "file_output.hpp"
#include <omp.h>
#include <unordered_set>
#include <map>
#include <random>


/*** PG-MSP parameters ***/
const unsigned int edge_size = 25;      // edge size
const unsigned int level_max = 22;      // max level
const unsigned int path_length = 16;    // path length

std::mt19937 mt(0), mt_(1);
std::uniform_real_distribution<> rnd(0, 1.0), rnd_(0, 1.0);

// local density max (global and local)
float local_density_max = 0;
float local_density_max_update = 0;
unsigned int id_local_density_max = 0;
unsigned int id_local_density_max_update = 0;


// path
std::deque<unsigned int> route;

// AkNN
std::vector<std::pair<float, unsigned int>> knn_list;

// pool
std::unordered_set<unsigned int> pool;


// greedy decision of dependent point
void get_dependent_point(data* dat) {

    // candidate list
    std::pair<float, unsigned int> candidate = {FLT_MAX, dat->identifier};

    // termination flag
    bool flag = 0;

    // get start node
    const unsigned int start_node = 0;

    // queue for NN & range search
    std::deque<unsigned int> queue_nn, queue_range;

    // compute distance
    float distance = compute_distance(&dataset[start_node], dat);

    // update candidate
    if (dataset[start_node].flag_active) {
        if (dataset[start_node].local_density > dat->local_density) candidate = {distance, start_node};
        if (dataset[start_node].local_density < dat->local_density) {
            if (dataset[start_node].dependent_distance > distance) dataset[start_node].dependent_distance = distance;
        }
    }

    // init queue
    queue_nn.push_back(start_node);
    if (distance <= cutoff * 2.5) queue_range.push_back(start_node);

    // init threshold for NN
    float threshold = distance;

    // set current result
    unsigned int id_min = start_node;

    // mark as visit
    std::unordered_set<unsigned int> visit;
    visit.insert(start_node);
    
    /*** ANN search ***/
    while (queue_nn.size() > 0) {

        // get the top
        const unsigned int id = queue_nn[0];

        // pop the top
        queue_nn.pop_front();

        // set current result
        id_min = id;

        // graph traversal
        for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

            // get id
            const unsigned int id_ = dataset[id].edges[i];

            // visit check
            if (visit.find(id_) == visit.end()) {

                // mark as visit
                visit.insert(id_);

                // distance computation
                distance = compute_distance(&dataset[id_], dat);

                // candidate update
                if (dataset[id_].flag_active) {
                    if (dataset[id_].local_density > dat->local_density) {
                        if (distance <= cutoff * 2.0) {

                            // update flag
                            flag = 1;

                            // update dependecy
                            dat->dependent_distance = distance;
                            dat->dependent_point_id = id_;

                            break;
                        }

                        if (candidate.first > distance) candidate = {distance, id_};
                    }

                    // update dependecy
                    if (dataset[id_].local_density < dat->local_density) {
                        if (dataset[id_].dependent_distance > distance) dataset[id_].dependent_distance = distance;
                    }
                }

                // NN update case
                if (distance < threshold) {
                    threshold = distance;
                    id_min = id_;
                }

                // queue update
                if (distance <= cutoff * 2.5) queue_range.push_back(id_);
            }
        }

        if (flag) break;

        // queue & route update
        if (id_min != id) {
            queue_nn.push_back(id_min);
            route.push_front(id_min);
        }
    }

    if (flag == 0) {

        // init visited nodes
        visit.clear();

        // vector for verified points
        std::vector<unsigned int> verification;

        // init queue
        if (threshold <= cutoff * 2.5) {

            // mark as visit
            visit.insert(id_min);

            // store point that needs verification
            if (dataset[id_min].member.size() > 0) verification.push_back(id_min);
        }

        /*** range search ***/
        while (queue_range.size() > 0 && flag == 0) {

            // get the top
            const unsigned int id = queue_range[0];

            // pop the top
            queue_range.pop_front();

            // graph traversal
            for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

                // get id
                const unsigned int id_ = dataset[id].edges[i];

                // visit check
                if (visit.find(id_) == visit.end()) {

                    // mark as visit
                    visit.insert(id_);

                    // distance computation
                    const float distance = compute_distance(&dataset[id_], dat);

                    // candidate update
                    if (dataset[id_].flag_active) {
                        if (dataset[id_].local_density > dat->local_density) {
                            if (distance <= cutoff * 2.0) {

                                // update dependency
                                dat->dependent_distance = threshold;
                                dat->dependent_point_id = id_;

                                // update flag
                                flag = 1;

                                break;
                            }
                            else {
                                if (candidate.first > distance) candidate = {distance, id_};
                            }
                        }

                        // update dependency
                        if (dataset[id_].local_density < dat->local_density) {
                            if (dataset[id_].dependent_distance > distance) dataset[id_].dependent_distance = distance;
                        }
                    }

                    // update queue
                    if (flag == 0) {
                        if (distance <= cutoff * 2.5) {

                            queue_range.push_back(id_);

                            // store point that needs verification
                            if (dataset[id_].member.size() > 0) verification.push_back(id_);
                        }
                    }
                }
            }
        }

        /*** verification ***/
        if (flag == 0) {
                
            std::vector<std::pair<float, unsigned int>> dep(thread_num);
            for (unsigned int i = 0; i < thread_num; ++i) dep[i] = {FLT_MAX, dat->identifier};

            #pragma omp parallel num_threads(thread_num)
            {
                for (unsigned int i = 0; i < verification.size(); ++i) {

                    // get id
                    const unsigned int id = verification[i];

                    #pragma omp for schedule(static)
                    for (unsigned int j = 0; j < dataset[id].member.size(); ++j) {

                        // get id_
                        const unsigned int id_ = dataset[id].member[j];

                        if (flag == 0) {
                            if (dataset[id_].flag_active) {

                                // distance computation
                                const float distance = compute_distance(dat, &dataset[id_]);

                                // dependecy update (local)
                                if (dataset[id_].local_density > dat->local_density) {

                                    if (distance <= 2.0 * cutoff) {

                                        flag = 1;
                                        dep[omp_get_thread_num()] = {distance, id_};
                                    }
                                    else {
                                        if (dep[omp_get_thread_num()].first > distance) dep[omp_get_thread_num()] = {distance, id_};
                                    }
                                }

                                // update dependency
                                if (dataset[id_].local_density < dat->local_density) {
                                    if (dataset[id_].dependent_distance > distance) dataset[id_].dependent_distance = distance;
                                }
                            }
                        }    
                    }
                }            
            }

            // reduction
            for (unsigned int i = 0; i < thread_num; ++i) {

                if (dat->dependent_distance > dep[i].first) {
                    dat->dependent_distance = dep[i].first;
                    dat->dependent_point_id = dep[i].second;
                }
            }

            // refinement
            if (dat->dependent_distance > cutoff * 2.0) {

                if (candidate.first < dat->dependent_distance) {
                    dat->dependent_distance = candidate.first;
                    dat->dependent_point_id = candidate.second;
                }
            }
        }
    }
}

// scan for finding dependent point
void get_dependent_point_scan(data* dat) {

    // init dependecy
    if (dat->local_density >= dataset[dat->dependent_point_id].local_density) {
        dat->dependent_distance = compute_distance(dat, &dataset[id_local_density_max]);
        dat->dependent_point_id = id_local_density_max;
    }

    // init local info.
    std::vector<std::pair<float, unsigned int>> dep(thread_num);
    for (unsigned int i = 0; i < thread_num; ++i) dep[i] = {dat->dependent_distance, dat->identifier};

    // make vector
    std::vector<unsigned int> pool_vec;
    auto it = pool.begin();
    while (it != pool.end()) {
        pool_vec.push_back(*it);
        ++it;
    }

    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for schedule(static)
        for (unsigned int i = 0; i < pool_vec.size(); ++i) {

            // get id
            const unsigned int id = pool_vec[i];

            // distance computation
            const float distance = compute_distance(dat, &dataset[id]);

            if (dat->local_density < dataset[id].local_density) {

                if (dep[omp_get_thread_num()].first > distance) {
                    dep[omp_get_thread_num()].first = distance;
                    dep[omp_get_thread_num()].second =dat->identifier;
                }
            }
            else {

                if (dataset[id].dependent_distance > distance) {
                    dataset[id].dependent_distance = distance;
                    dataset[id].dependent_point_id = dat->identifier;
                }
            }
        }
    }

    // reduction
    for (unsigned int i = 0; i < thread_num; ++i) {
        
        if (dat->dependent_distance > dep[i].first) {
            dat->dependent_distance = dep[i].first;
            dat->dependent_point_id = dep[i].second;
        }
    }

    //if (dat->dependent_distance > cutoff * 2.0) pool.insert(dat->identifier);
    pool.insert(dat->identifier);
}

// local density update (insertion case)
void update_local_density_insertion(std::vector<unsigned int> &dataset_rho_update, std::deque<data*> &dataset_active, data* dat) {

    // init
    local_density_max_update = 0;
    id_local_density_max_update = dat->identifier;
    dataset_rho_update.clear();

    // init parent
    dat->parent = dat->identifier;

    // flag update
    dat->flag_update = 1;
    dat->flag_active = 1;

    // init dep. point
    dat->dependent_point_id = dat->identifier;

    if (dat->identifier > 0) {

        // get start node
        const unsigned int start_node = 0;

        // init route
        route.push_back(start_node);

        // queue for NN & range search
        std::deque<unsigned int> queue_nn, queue_range;

        // vector for verified points
        std::vector<unsigned int> verification;

        // compute distance
        float distance = compute_distance(&dataset[start_node], dat);

        // init queue
        queue_nn.push_back(start_node);
        if (distance <= cutoff * 1.5) queue_range.push_back(start_node);

        // init threshold for NN
        float threshold = distance;

        // set current result
        unsigned int id_min = start_node;

        // mark as visit
        std::unordered_set<unsigned int> visit;
        visit.insert(start_node);

        // init candidate of kNN
        knn_list.push_back({distance, start_node});

        /*** ANN search ***/
        while (queue_nn.size() > 0) {

            // get the top
            const unsigned int id = queue_nn[0];

            // pop the top
            queue_nn.pop_front();

            // set current result
            id_min = id;

            // graph traversal
            for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

                // get id
                const unsigned int id_ = dataset[id].edges[i];

                // visit check
                if (visit.find(id_) == visit.end()) {

                    // mark as visit
                    visit.insert(id_);

                    // distance computation
                    distance = compute_distance(&dataset[id_], dat);

                    // kNN candidate update
                    knn_list.push_back({distance, id_});

                    // NN update case
                    if (distance < threshold) {
                        threshold = distance;
                        id_min = id_;
                    }

                    // queue update
                    if (distance <= cutoff * 1.5) queue_range.push_back(id_);
                }
            }

            // queue & route update
            if (id_min != id) {
                queue_nn.push_back(id_min);
                route.push_front(id_min);
            }
        }

        // init visited nodes
        visit.clear();

        // init queue
        if (threshold <= cutoff * 1.5) {

            if (dataset[id_min].flag_active) {

                if (threshold <= cutoff) {

                    // update local density
                    dat->local_density += 1.0;
                    dataset[id_min].local_density += 1.0;

                    // update max of local density
                    if (dataset[id_min].local_density > local_density_max_update) {
                        local_density_max_update = dataset[id_min].local_density;
                        id_local_density_max_update = id_min;
                    }
                    if (dataset[id_min].local_density > local_density_max) {
                        local_density_max = dataset[id_min].local_density;
                        id_local_density_max = id_min;
                    }

                    // density update flag
                    dataset[id_min].flag_update = 1;
                    dat->flag_update = 1;

                    // update neighbor set
                    dataset_rho_update.push_back(id_min);
                }
            }

            // mark as visit
            visit.insert(id_min);

            // store point that needs verification
            if (dataset[id_min].member.size() > 0) verification.push_back(id_min);

            // member update
            if (threshold <= cutoff / 2.0 && rnd(mt) <= 0.995) {

                dat->parent = id_min;
                dat->flag_pivot = 0;
            }
        }

        /*** range search ***/
        while (queue_range.size() > 0) {

            // get the top
            const unsigned int id = queue_range[0];

            // pop the top
            queue_range.pop_front();

            // graph traversal
            for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

                // get id
                const unsigned int id_ = dataset[id].edges[i];

                // visit check
                if (visit.find(id_) == visit.end()) {

                    // mark as visit
                    visit.insert(id_);

                    // distance computation
                    const float distance = compute_distance(&dataset[id_], dat);

                    // store kNN candidate
                    knn_list.push_back({distance, id_});

                    if (distance <= cutoff * 1.5) {

                        // update queue
                        queue_range.push_back(id_);

                        // set temporal distance
                        dataset[id_].distance_temp = distance;

                        // update local density
                        if (distance <= cutoff && dataset[id_].flag_active == 1) {

                            dataset[id_].local_density += 1.0;
                            dat->local_density += 1.0;

                            // update max of local density
                            if (dataset[id_].local_density > local_density_max) {
                                local_density_max = dataset[id_].local_density;
                                id_local_density_max = id_;
                            }
                            if (dataset[id_].local_density > local_density_max_update) {
                                local_density_max_update = dataset[id_].local_density;
                                id_local_density_max_update = id_;
                            }

                            dataset_rho_update.push_back(id_);
                        }

                        // store point that needs verification
                        if (dataset[id_].member.size() > 0) verification.push_back(id_);
                    }
                }
            }
        }

        /*** verification ***/
        float local_density = dat->local_density;
        std::vector<std::pair<float, unsigned int>> max_update(thread_num);
        std::vector<std::vector<unsigned int>> rho_update(thread_num);

        #pragma omp parallel num_threads(thread_num)
        {
            for (unsigned int i = 0; i < verification.size(); ++i) {

                // get id
                const unsigned int id = verification[i];

                bool f = 0;
                if (dataset[id].distance_temp <= cutoff / 2.0) f = 1;

                #pragma omp for schedule(static) reduction(+:local_density)
                for (unsigned int j = 0; j < dataset[id].member.size(); ++j) {

                    // get id_
                    const unsigned int id_ = dataset[id].member[j];

                    if (f) {

                        if (dataset[id_].flag_active) {

                            // local density update
                            local_density = local_density + 1.0;
                            dataset[id_].local_density = dataset[id_].local_density + 1.0;

                            // flag update
                            dataset[id_].flag_update = 1;

                            rho_update[omp_get_thread_num()].push_back(id_);

                            // max update
                            if (max_update[omp_get_thread_num()].first < dataset[id_].local_density) max_update[omp_get_thread_num()] = {dataset[id_].local_density,id_};
                        }
                    }
                    else {

                        if (dataset[id_].flag_active) {

                            const float distance = compute_distance(dat, &dataset[id_]);

                            if (distance <= cutoff) {

                                // local density update
                                local_density = local_density + 1.0;
                                dataset[id_].local_density = dataset[id_].local_density + 1.0;

                                // flag update
                                dataset[id_].flag_update = 1;

                                rho_update[omp_get_thread_num()].push_back(id_);

                                // max update
                                if (max_update[omp_get_thread_num()].first < dataset[id_].local_density) max_update[omp_get_thread_num()] = {dataset[id_].local_density,id_};
                            }
                        }
                    }
                }

                // init
                dataset[id].distance_temp = FLT_MAX;
            }            
        }

        // merge
        dat->local_density = local_density;

        // store id of the new object
        dataset_rho_update.push_back(dat->identifier);

        // update local_density_max_update
        for (unsigned int i = 0; i < thread_num; ++i) {

            if (max_update[i].first > local_density_max_update) {

                local_density_max_update = max_update[i].first;
                id_local_density_max_update = max_update[i].second;
            }
            if (max_update[i].first > local_density_max) {

                local_density_max = max_update[i].first;
                id_local_density_max = max_update[i].second;
            }

            for (unsigned int j = 0; j < rho_update[i].size(); ++j) dataset_rho_update.push_back(rho_update[i][j]);
        }
        if (dat->local_density > local_density_max_update) {
            local_density_max_update = dat->local_density;
            id_local_density_max_update = dat->identifier;
        }
        if (dat->local_density > local_density_max) {
            local_density_max = dat->local_density;
            id_local_density_max = dat->identifier;
        }

        // update avg. local density
        local_density_avg += dat->local_density;
    }
}

// graph update
void update_graph(std::vector<unsigned int> &dataset_rho_update, data* dat) {

    if (dat->flag_pivot) {

        // set itself as parent
        dat->parent = dat->identifier;

        // hash for edge
        std::unordered_set<unsigned int> edges;

        // sort
        std::sort(knn_list.begin(), knn_list.end());

        // connect to AkNN
        for (unsigned int i = 0; i < knn_list.size(); ++i) {

            const unsigned int id = knn_list[i].second;

            if (edges.find(id) == edges.end()) {

                dataset[id].edges.push_back(dat->identifier);
                dat->edges.push_back(id);
                edges.insert(id);
            }

            if (edges.size() == edge_size) break;
        }

        // making a skip monotonic path
        unsigned int id = route[0];

        if (route.size() < path_length) {

            for (unsigned int j = 0; j < route.size(); ++j) {

                if (dataset[route[j]].level == dat->level) {
                    id = route[j];
                    break;
                }
            }
        }
        else {

            unsigned int size = 2;
            while (1) {
                if ((route.size() / size) < path_length) break;
                size *= 2;
            }
            id = route[route.size() / size];
        }

        if (id != 0)  {

            if (edges.find(id) == edges.end()) {
                dataset[id].edges.push_back(dat->identifier);
                dat->edges.push_back(id);
            }
        }
    }
    else {

        // member update
        dataset[dat->parent].member.push_back(dat->identifier);
    }

    length += route.size();
}

// dependent point update (insertion case)
void update_dependent_point_insertion(std::vector<unsigned int> &dataset_rho_update, std::deque<data*> &dataset_active, data* dat) {

    bool f = 0;

    std::vector<std::unordered_set<unsigned int>> local_pool(thread_num);

    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for schedule(static) reduction(+:cnt)
        for (unsigned int i = 0; i < dataset_rho_update.size(); ++i) {

            // get id
            const unsigned int id = dataset_rho_update[i];

            if (dataset[id].identifier == id_local_density_max) {

                // set itself as dep. point
                dataset[id].dependent_distance = FLT_MAX;
                dataset[id].dependent_point_id = id;
            }
            else if (dataset[id].identifier == id_local_density_max_update) {

                if (dataset[id].identifier == dat->identifier) {

                    // sort AkNN list
                    if (dataset[id].flag_pivot == 0) std::sort(knn_list.begin(), knn_list.end());

                    for (unsigned int j = 0; j < knn_list.size(); ++j) {

                        // get id_
                        const unsigned int id_ = knn_list[j].second;

                        // set dep. point
                        if (dataset[id].local_density < dataset[id_].local_density) {

                            if (dataset[id_].flag_active) {
                                dataset[id].dependent_distance = knn_list[j].first;
                                dataset[id].dependent_point_id = id_;
                                break;
                            }
                        }
                    }
                }
                else {

                    // get dependent point
                    const unsigned int id_ = dataset[id].dependent_point_id;

                    if (dataset[id_].local_density < dataset[id].local_density) {

                        // init
                        dataset[id].dependent_point_id = id;
                        dataset[id].dependent_distance = FLT_MAX;

                        f = 1;
                    }
                    else if (dataset[id].dependent_distance > cutoff * 2.0) {
                        if (dataset[id].local_density > 5.0) f = 1;
                    }
                }
            }
            else {

                local_pool[omp_get_thread_num()].insert(id);

                // get dependent point
                const unsigned int id_ = dataset[id].dependent_point_id;

                // update case
                bool flag = 0;
                if (dataset[id].local_density > dataset[id_].local_density) {
                    flag = 1;
                }
                else if (dataset[id].dependent_distance > cutoff * 2) {
                    flag = 1;
                }

                // update dep. point
                if (flag) {
                    dataset[id].dependent_point_id = id_local_density_max_update;
                    dataset[id].dependent_distance = 2 * cutoff;
                }
            }

            // init flag
            dataset[id].flag_update = 0;
        }
    }

    if (f) {

        ++cnt;
        get_dependent_point_scan(&dataset[id_local_density_max_update]);
    }

    // reduction
    for (unsigned int i = 0; i < thread_num; ++i) {

        auto it = local_pool[i].begin();
        while (it != local_pool[i].end()) {
            if (rnd_(mt_) <= 0.01) pool.erase(*it);
            ++it;
        }
    }

    pool_size = pool.size();

    // init
    route.clear();
    knn_list.clear();

    // init
    id_local_density_max_update = 0;
}

// local density update (deletion case)
void update_local_density_deletion(std::deque<data*> &dataset_active, data* dat) {

    // remove from pool
    pool.erase(dat->identifier);

    // get start node
    const unsigned int start_node = 0;

    // queue for NN & range search
    std::deque<unsigned int> queue_nn, queue_range;

    // vector for verified points
    std::vector<unsigned int> verification;

    // compute distance
    float distance = compute_distance(&dataset[start_node], dat);

    // init queue
    queue_nn.push_back(start_node);
    if (distance <= cutoff * 1.5) queue_range.push_back(start_node);

    // init threshold for NN
    float threshold = distance;

    // set current result
    unsigned int id_min = start_node;

    // mark as visit
    std::unordered_set<unsigned int> visit;
    visit.insert(start_node);

    /*** ANN search ***/
    while (queue_nn.size() > 0) {

        // get the top
        const unsigned int id = queue_nn[0];

        // pop the top
        queue_nn.pop_front();

        // set current result
        id_min = id;

        // graph traversal
        for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

            // get id
            const unsigned int id_ = dataset[id].edges[i];

            // visit check
            if (visit.find(id_) == visit.end()) {

                // mark as visit
                visit.insert(id_);

                // distance computation
                distance = compute_distance(&dataset[id_], dat);

                // NN update case
                if (distance < threshold) {
                    threshold = distance;
                    id_min = id_;
                }

                // queue update
                if (distance <= cutoff * 1.5) queue_range.push_back(id_);
            }
        }

        // queue update
        if (id_min != id) queue_nn.push_back(id_min);
    }    

    // init queue
    if (threshold <= cutoff * 1.5) {

        if (dataset[id_min].flag_active) {

            if (threshold <= cutoff) {

                // update local density
                dataset[id_min].local_density -= 1.0;

                // density update flag
                dataset[id_min].flag_update = 1;
            }
        }    
            
        // mark as visit
        visit.insert(id_min);

        // store point that needs verification
        if (dataset[id_min].member.size() > 0) verification.push_back(id_min);
    }

    /*** range search ***/
    while (queue_range.size() > 0) {

        // get the top
        const unsigned int id = queue_range[0];

        // pop the top
        queue_range.pop_front();

        // graph traversal
        for (unsigned int i = 0; i < dataset[id].edges.size(); ++i) {

            // get id
            const unsigned int id_ = dataset[id].edges[i];

            // visit check
            if (visit.find(id_) == visit.end()) {

                // mark as visit
                visit.insert(id_);

                // distance computation
                const float distance = compute_distance(&dataset[id_], dat);

                if (distance <= cutoff * 1.5) {

                    // update queue
                    queue_range.push_back(id_);

                    // set temporal distance
                    dataset[id_].distance_temp = distance;

                    // update local density
                    if (distance <= cutoff && dataset[id_].flag_active == 1) {

                        dataset[id_].local_density -= 1.0;
                        dataset[id_].flag_update = 1;
                    }

                    // store point that needs verification
                    if (dataset[id_].member.size() > 0) verification.push_back(id_);
                }
            }
        }
    }

    // verification
    #pragma omp parallel num_threads(thread_num)
    {
        for (unsigned int i = 0; i < verification.size(); ++i) {

            // get id
            const unsigned int id = verification[i];

            bool f = 0;
            if (dataset[id].distance_temp <= cutoff / 2.0) f = 1;

            #pragma omp for schedule(static)
            for (unsigned int j = 0; j < dataset[id].member.size(); ++j) {

                // get id_
                const unsigned int id_ = dataset[id].member[j];

                if (f) {

                    if (dataset[id_].flag_active) {

                        // local density update
                        dataset[id_].local_density -= 1.0;

                        // flag update
                        dataset[id_].flag_update = 1;
                    }
                }
                else {

                    if (dataset[id_].flag_active) {

                        const float distance = compute_distance(dat, &dataset[id_]);

                        if (distance <= cutoff) {

                            // local density update
                            dataset[id_].local_density -= 1.0;

                            // flag update
                            dataset[id_].flag_update = 1;
                        }
                    }
                }
            }

            // init
            dataset[id].distance_temp = FLT_MAX;
        }            
    }

    // update avg. local density
    local_density_avg += dat->local_density;
}

// dependent point update (deletion case)
void update_dependent_point_deletion(std::deque<data*> &dataset_active, data* dat) {

    // init local density
    dat->local_density = 0;

    // local array for object that needs dependency update
    std::vector<std::vector<unsigned int>> dependecy_update_local(thread_num);

    // local array for local_density_max
    std::vector<std::pair<float, unsigned int>> local_density_max_local(thread_num);

    // init local_density_max
    local_density_max = 0;

    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for schedule(static)
        for (unsigned int i = 0; i < dataset_active.size(); ++i) {

            if (dataset_active[i]->flag_update == 0) {

                // get id of dependent point
                const unsigned int id = dataset_active[i]->dependent_point_id;

                if (dataset_active[i]->local_density >= dataset[id].local_density) dependecy_update_local[omp_get_thread_num()].push_back(i);
            }
            else {

                // init flag
                dataset_active[i]->flag_update = 0;
            }

            // update local max local density
            if (local_density_max_local[omp_get_thread_num()].first < dataset_active[i]->local_density) local_density_max_local[omp_get_thread_num()] = {dataset_active[i]->local_density, dataset_active[i]->identifier};
        }
    }

    // reduction
    std::vector<unsigned int> dependecy_update;
    for (unsigned int i = 0; i < thread_num; ++i) {

        // merge dependecy_update_local
        for (unsigned int j = 0; j < dependecy_update_local[i].size(); ++j) dependecy_update.push_back(dependecy_update_local[i][j]);

        // update local_density_max
        if (local_density_max < local_density_max_local[i].first) {
            local_density_max = local_density_max_local[i].first;
            id_local_density_max = local_density_max_local[i].second;
        }
    }

    // update dependecy
    #pragma omp parallel num_threads(thread_num)
    {
        #pragma omp for schedule(dynamic)
        for (unsigned int i = 0; i < dependecy_update.size(); ++i) {
            dataset_active[dependecy_update[i]]->update_dependent_point(id_local_density_max);
        }
    }

    for (unsigned int i = 0; i < dependecy_update.size(); ++i) {

        // get id
        const unsigned int id = dataset_active[dependecy_update[i]]->identifier;

        if (dataset[id].dependent_distance > cutoff * 2.0) {
            ++cnt;
            get_dependent_point(&dataset[id]);
        }
    }
}