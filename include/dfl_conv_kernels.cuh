#include<string>
#include<vector>
#include<iostream>
#include<fstream>

using namespace std;

template <typename T>
bool input_load(const string &path, vector<T> &data){
    ifstream file(path , ios::binary | ios::ate);

    if(!file.is_open())
    {
        cerr<<"file can not open "<<path<<endl;
        return false;
    }

    file.seekg(0, ios::beg);

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<streamsize>(data.size() * sizeof(T))
    );

    if(!file){
        cerr<<"file is empty"<<endl;
        return false;
    }

    return true;
}
template <typename T>
bool output_write(const string &path, vector<T> &data){
    ofstream file(path , ios::binary | ios::trunc);

    if(!file.is_open())
    {
        cerr<<"can not create file "<<path<<endl;
        return false;
    }

    file.write(
        reinterpret_cast<char*>(data.data()),
        static_cast<streamsize>(data.size() * sizeof(T))
    );

    return static_cast<bool>(file);
}