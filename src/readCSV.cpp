// Reads from csv into 2d array
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

std::vector<double> readCSV(const std::string& filepath)
{
    std::vector<double> data; 
    std::ifstream file(filepath);
    std::string line;

    if (! file.is_open()){
        cerr << "Error opening file!" << endl;
        return data;
    }

    //
    while (std::getline(file, line)){
        stringstream ss(line);
        string cell; 
        int col = 0;
        while(std::getline(ss, cell, ',')) {
            data.push_back(std::stod(cell));
        };
    }

    file.close();

    return data;

}