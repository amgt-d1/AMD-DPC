#include "pg_msp.hpp"


int main() {

    // input parameter
    input_parameter();

    // input dataset
    input_data();

    // print the current time
    get_current_time();

    std::cout << " --------------------\n";
	std::cout << " dataset id: " << dataset_id << "\n";
	std::cout << " dataset cardinality: " << dataset.size() << "\n";
	std::cout << " dataset dimensionality: " << dimensionality << "\n";
    std::cout << " deletion rate: " << deletion_rate << "\n";
    std::cout << " cutoff distance: " << cutoff << "\n";
    std::cout << " number of threads: " << thread_num << "\n";
	std::cout << " --------------------\n\n";


    // random generator for insertion & deletion
    std::mt19937 mt(0);
    std::uniform_real_distribution<> rnd(0,1.0);

    // random generator for local density
    std::mt19937 mt_(1);
    std::uniform_real_distribution<> rnd_rho(0,0.999);

    // idx of insertion & deletion
    unsigned int idx_insertion = 0;
    unsigned int idx_deletion = 0;
    unsigned int counter = 0;

    // a set of inserted objects (not deleted)
    std::deque<data*> dataset_active;

    // get initial memory
    double memory_init = process_mem_usage();


    /****** start dynamic clustering  ******/
    while (1) {

        // determine insertion or deletion; 1: insertion, 0: deletion
        bool flag = 1;
        if (idx_insertion >= freq) {
            if (rnd(mt) <= deletion_rate) flag = 0;
        }

        if (flag) {

            /**** main update start ****/
            start = std::chrono::system_clock::now();

            // add a random value into local density
            dataset[idx_insertion].local_density = rnd_rho(mt_);

            // append new object
            dataset_active.push_back(&dataset[idx_insertion]);

            // local density computation
            std::vector<unsigned int> dataset_rho_update;   // an element is the identifier
            update_local_density_insertion(dataset_rho_update, dataset_active, &dataset[idx_insertion]);

            end = std::chrono::system_clock::now();
            local_density_update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            start = std::chrono::system_clock::now();

            // index update
            update_graph(dataset_rho_update, &dataset[idx_insertion]);

            end = std::chrono::system_clock::now();
            index_update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            start = std::chrono::system_clock::now();

            // dependent point update
            update_dependent_point_insertion(dataset_rho_update, dataset_active, &dataset[idx_insertion]);

            end = std::chrono::system_clock::now();
            dependent_point_update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // increment idx
            ++idx_insertion;
        }
        else {

            // get deleted data
            data* dat = &dataset[idx_deletion];

            // update active flag
            dat->flag_active = 0;

            /**** main update start ****/
            start = std::chrono::system_clock::now();

            // delete the front data
            dataset_active.pop_front();

            // local density computation
            update_local_density_deletion(dataset_active, dat);

            end = std::chrono::system_clock::now();
            local_density_update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            start = std::chrono::system_clock::now();

            // dependent point update
            update_dependent_point_deletion(dataset_active, dat);

            end = std::chrono::system_clock::now();
            dependent_point_update_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // increment idx
            ++idx_deletion;
        }

        // increment counter
        ++counter;

        // if no more object cannot be inserted, break after output
        if (idx_insertion == dataset.size() - 1) {    

            // get current memory [MB]
            memory = process_mem_usage() - memory_init;

            // output final result
            output_result(counter, 0);

            break;
        }

        if (counter % freq == 0) {

            // print
            std::cout << " " << counter << " updates over\n";
 
            // get current memory [MB]
            memory = process_mem_usage() - memory_init;

            // output intermediate result
            output_result(counter, 1);
        }

        if (counter % 100000 == 0) output_statistics(counter, dataset_active);
    }

    return 0;
}