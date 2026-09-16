#include<string>
#include<vector>
#include<iostream>
#include<fstream>

using namespace std;

bool input_load(const string &path, vector<float> &data){
    std::ifstream file(path , std::ios::binary | std::ios::ate);

    if(!file.is_open())
    {
        cerr<<"file can not open "<<path<<endl;
        return false;
    }

    file.seekg(0, std::ios::beg);

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float))
    );

    if(!file){
        cerr<<"file is empty"<<endl;
        return false;
    }

    return true;
}

bool output_write(const std::string &path, vector<float> &data){
    std::ofstream file(path , std::ios::binary | std::ios::trunc);

    if(!file.is_open())
    {
        cerr<<"can not create file "<<path<<endl;
        return false;
    }

    file.write(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float))
    );

    return static_cast<bool>(file);
}