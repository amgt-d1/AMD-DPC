#include <vector>
#include "file_input.hpp"
#include <unordered_map>
#include <cfloat>
#include <algorithm>
#include <deque>
#include <math.h>


// declaration of data
class data;

// definition of dataset
std::vector<data> dataset;

// compute distance
float compute_distance(const data* p, const data* q);


// definition of data class
class data {

public:

	// for numeric data
	std::vector<float> pt;

	// identifier
	unsigned int identifier = 0;

	// local density
	float local_density = 0;

	// dependent point
	unsigned int dependent_point_id = 0;

	// dependent distance
	float dependent_distance = FLT_MAX;


	// edges of the proximity graph
	std::vector<unsigned int> edges;

	// node level
	unsigned int level = 0;

	// active flag
	bool flag_active = 0;

	// pivot flag
	bool flag_pivot = 1;

	// local density update flag
	bool flag_update = 0;

	// parent node
	unsigned int parent = 0;

	// member
	std::vector<unsigned int> member;

	// temp distance
	float distance_temp = FLT_MAX;


	/***************/
	/* constructor */
	/***************/

	// standard
	data() {}

	// with identifier
	data(const unsigned int id)
	{
		identifier = id;
		dependent_point_id = id;
	}

	/*******************/
	/* member function */
	/*******************/

	// point update
	void update_pt(const std::vector<float>& point) { for (unsigned int i = 0; i < point.size(); ++i) pt.push_back(point[i]); }

	// dependent point update
	void update_dependent_point(const unsigned int id_local_density_max)
	{
		// init dependency
		dependent_distance = FLT_MAX;
		dependent_point_id = identifier;

		bool f = 0;

		if (identifier != id_local_density_max)
		{
			// it is not pivot
			if (identifier != parent)
			{
				if (dataset[parent].flag_active)
				{
					if (dataset[parent].local_density > local_density)
					{
						dependent_point_id = parent;
						dependent_distance = cutoff / 2.0;

						f = 1;
					}
				}
			}

			if (f == 0)
			{
				std::vector<unsigned int> verification;

				for (unsigned int i = 0; i < dataset[parent].edges.size(); ++i)
				{
					// get id
					const unsigned int id = dataset[parent].edges[i];

					// update dependency
					if (dataset[id].flag_active)
					{
						if (dataset[id].local_density > local_density)
						{
							const float distance = compute_distance(this, &dataset[id]);
							if (distance <= cutoff * 2.0)
							{
								f = 1;
								dependent_distance = distance;
								dependent_point_id = id;
								break;
							}
							else
							{
								if (dataset[id].member.size() > 0) verification.push_back(id);

								if (distance < dependent_distance)
								{
									dependent_distance = distance;
									dependent_point_id = id;
								}
							}
						}
					}
				}

				if (f == 0)
				{
					for (unsigned int i = 0; i < verification.size(); ++ i)
					{
						for (unsigned int j = 0; j < dataset[verification[i]].member.size(); ++j)
						{
							const unsigned int id = dataset[verification[i]].member[j];

							if (dataset[id].flag_active)
							{
								if (dataset[id].local_density > local_density)
								{
									const float distance = compute_distance(this, &dataset[id]);

									if (distance <= cutoff * 2.0)
									{
										f = 1;
										dependent_distance = distance;
										dependent_point_id = id;
										break;
									}
									else
									{
										if (distance < dependent_distance)
										{
											dependent_distance = distance;
											dependent_point_id = id;
										}
									}
								}
							}
						}

						if (f) break;
					}
				}
			}
		}	
	}
};


// input data
void input_data()
{
	// id variable
	unsigned int id = 0;

	// point coordinates variable
    data dat;
    dat.pt.resize(dimensionality);

    // dataset input
	std::string f_name = "../dataset/";
	if (dataset_id == 1) f_name += "household-7d.txt";
	if (dataset_id == 2) f_name += "pamap2-51d.txt";
	if (dataset_id == 3) f_name += "gas-18d.txt";
	if (dataset_id == 4) f_name += "mirai-115d.txt";

    // file input
	std::ifstream ifs_file(f_name);
    std::string full_data;

	// error check
	if (ifs_file.fail())
	{
		std::cout << " data file does not exist." << std::endl;
		std::exit(0);
	}

	while (std::getline(ifs_file, full_data))
	{
		std::string meta_info;
		std::istringstream stream(full_data);
		std::string type = "";

		for (unsigned int i = 0; i < dimensionality; ++i)
		{
			std::getline(stream, meta_info, ',');
			std::istringstream stream_(meta_info);
			long double val = std::stold(meta_info);
			if (val < 0.0000001) val = 0;
			dat.pt[i] = val;
		}

		// update id
		dat.identifier = id;

		// update dependent point id
		dat.dependent_point_id = id;

		// insert into dataset
		dataset.push_back(dat);

		// increment identifier
		++id;
	}
}

// compute distance
float compute_distance(const data* p, const data* q)
{
	float distance = 0;

	if (type == 0)
	{
		// euclidean (l2) distance
		for (unsigned int i = 0; i < dimensionality; ++i)
		{
			distance += powf(p->pt[i] - q->pt[i], 2.0);
		}
	}
	else if (type == 1)
	{
		// manhattan (l1) distance
		for (unsigned int i = 0; i < dimensionality; ++i)
		{
			distance += fabsf(p->pt[i] - q->pt[i]);
		}
	}

	return distance;
}