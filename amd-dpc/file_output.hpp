#include "data.hpp"
#include <chrono>
#include <unistd.h>


// frequency of output
const unsigned int freq = 10000;

// variable for time measure
std::chrono::system_clock::time_point start, end;

// computation time
double local_density_update_time = 0;
double dependent_point_update_time = 0;
double index_update_time = 0;
double update_time_total = 0;
long double update_time_total_local_density = 0;
long double update_time_total_dependent_point = 0;
long double update_time_total_index = 0;

// peak memory
double memory = 0;

// avg. local density
float local_density_avg = 0;

unsigned int length = 0;
unsigned int cnt = 0;
unsigned int pool_size = 0;

// compute memory usage
double process_mem_usage()
{
    double resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
        std::string ignore;
        std::ifstream ifs("/proc/self/stat", std::ios_base::in);
        ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    resident_set = rss * page_size_kb;

	return resident_set / 1000;
}

// result output
void output_result(const unsigned int counter, const bool flag)
{
    std::string f_name = "result/";
    if (dataset_id == 1) f_name += "1-household/";
	if (dataset_id == 2) f_name += "2-pamap2/";
	if (dataset_id == 3) f_name += "3-gas/";
    if (dataset_id == 4) f_name += "4-mirai/";

    f_name += "id(" + std::to_string(dataset_id) + ")_cutoff(" + std::to_string(cutoff) + ")_deletion_rate(" + std::to_string(deletion_rate) + ")_thread_num(" + std::to_string(thread_num) + ").csv";

    std::ofstream file;
    file.open(f_name.c_str(), std::ios::out | std::ios::app);

    if (file.fail())
    {
        std::cout << " cannot open the result file.\n";
        file.clear();
        return;
    }

    // increment update_time_total
    update_time_total += local_density_update_time + dependent_point_update_time + index_update_time;
    update_time_total_local_density += local_density_update_time;
    update_time_total_dependent_point += dependent_point_update_time;
    update_time_total_index += index_update_time;

    // output: counter, update time, memory
    if (flag == 1)
    {
        file
        << (local_density_update_time + dependent_point_update_time + index_update_time) / (freq * 1000) << ","
        << memory << "," 
        << local_density_update_time / (freq * 1000) << ","
        << dependent_point_update_time / (freq * 1000) << ","
        << index_update_time / (freq * 1000) << ","
        << local_density_avg << ","
        << cnt << ","
        << length << ","
        << pool_size << ","
        << "\n";
    }
    else
    {
        file
        << update_time_total / (1000 * counter) << ","
        << memory << ","
        << update_time_total_local_density / (1000 * counter) << ","
        << update_time_total_dependent_point / (1000 * counter) << ","
        << update_time_total_index / (counter * 1000) << ","
        << counter << "\n";
    }

    // init update_time
    local_density_update_time = 0;
    dependent_point_update_time = 0;
    index_update_time = 0;

    // init avg. local density
    local_density_avg = 0;
    cnt = 0;
    length = 0;
}
